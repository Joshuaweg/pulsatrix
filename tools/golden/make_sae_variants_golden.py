"""Writes golden values for FEAT-4's sparse autoencoders, written out in PyTorch from their papers
as pulsatrix implements them, for tests/sae_variants_test.cpp:

    python3 tools/golden/make_sae_variants_golden.py tests/fixtures/sae/sae_variants_golden.safetensors

batch.*: a Matryoshka BatchTopK SAE (Bussmann et al., arXiv 2412.06410 and 2503.17547). As
make_topk_sae_golden.py's TopK SAE, except that the batch's N*k largest activations are kept
wherever they fall, and the loss sums the reconstruction errors of the prefixes [0, 5), [0, 12)
and [0, 20), plus AuxK on the dead latents. Two Adam steps (lr 1e-2), each followed by unit decoder
rows; the inference threshold is the first batch's smallest kept activation, then a 0.99 moving
average.

jump.*: a JumpReLU SAE (Rajamanoharan et al., arXiv 2407.14435): z = pre * H(pre - θ),
pre = x W_enc + b_enc, θ = exp(log_threshold), loss mean((x̂ - x)²) + λ * mean over rows of the
active count, with the paper's straight-through estimators (rectangle kernel, bandwidth ε). Two
Adam steps, each followed by unit decoder rows with the scale moved into the encoder column, bias
and threshold.

Weights are (in, out) as pulsatrix stores them. Needs torch and safetensors.
"""

import math
import sys

import torch
from safetensors.torch import save_file

DIM, M, N = 6, 20, 9


def rect(u):
    return (u.abs() < 0.5).to(u.dtype)


class JumpReLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, theta, eps):
        ctx.save_for_backward(x, theta)
        ctx.eps = eps
        return x * (x > theta).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        x, theta = ctx.saved_tensors
        eps = ctx.eps
        gx = (x > theta).to(x.dtype) * g
        gtheta = (-(theta / eps) * rect((x - theta) / eps) * g).sum(0)
        return gx, gtheta, None


class StepFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, theta, eps):
        ctx.save_for_backward(x, theta)
        ctx.eps = eps
        return (x > theta).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        x, theta = ctx.saved_tensors
        eps = ctx.eps
        gtheta = (-(1.0 / eps) * rect((x - theta) / eps) * g).sum(0)
        return torch.zeros_like(x), gtheta, None


def project(w, g):
    return g - (g * w).sum(1, keepdim=True) / (w * w).sum(1, keepdim=True) * w


def batch_golden(t):
    K, K_AUX, ALPHA, PREFIXES = 3, 4, 1.0 / 32, [5, 12, 20]
    torch.manual_seed(7)
    w_enc = torch.randn(DIM, M) * 0.5
    b_enc = torch.randn(M) * 0.1
    w_dec = torch.randn(M, DIM)
    w_dec = w_dec / w_dec.norm(dim=1, keepdim=True)
    b_dec = torch.randn(DIM) * 0.2
    x = torch.randn(N, DIM)
    dead = torch.tensor([2, 6, 9, 15, 18], dtype=torch.int64)

    def loss_fn(we, be, wd, bd):
        pre = (x - bd) @ we + be
        vals, idx = pre.flatten().topk(N * K)
        kept = torch.relu(vals)
        z = torch.zeros(N * M).scatter(0, idx, kept).reshape(N, M)
        total = 0
        for p in PREFIXES:
            xh = z[:, :p] @ wd[:p] + bd
            mse = ((xh - x) ** 2).mean()
            total = total + mse
        e = (x - xh).detach()
        pre_dead = pre[:, dead]
        avals, aidx = pre_dead.topk(min(K_AUX, len(dead)), dim=1)
        z_aux = torch.zeros_like(pre_dead).scatter(1, aidx, torch.relu(avals))
        aux = (((z_aux @ wd[dead] - e) ** 2).sum(1) / (e ** 2).sum(1)).mean()
        min_kept = kept[kept > 0].min()
        return total + ALPHA * aux, mse, ALPHA * aux, min_kept.detach()

    params = [p.clone().requires_grad_(True) for p in (w_enc, b_enc, w_dec, b_dec)]
    total, mse, aux, _ = loss_fn(*params)
    total.backward()
    grads = [p.grad.clone() for p in params]
    grads[2] = project(params[2].detach(), grads[2])
    for k, v in {"w_enc": w_enc, "b_enc": b_enc, "w_dec": w_dec, "b_dec": b_dec, "x": x, "dead": dead,
                 "loss": total.detach().reshape(1), "reconstruction": mse.detach().reshape(1), "aux": aux.detach().reshape(1),
                 "grad_w_enc": grads[0], "grad_b_enc": grads[1], "grad_w_dec": grads[2], "grad_b_dec": grads[3]}.items():
        t["batch." + k] = v

    params = [p.detach().clone().requires_grad_(True) for p in (w_enc, b_enc, w_dec, b_dec)]
    opt = torch.optim.Adam(params, lr=1e-2)
    theta = None
    for _ in range(2):
        opt.zero_grad()
        total, _, _, min_kept = loss_fn(*params)
        total.backward()
        params[2].grad = project(params[2].detach(), params[2].grad)
        opt.step()
        with torch.no_grad():
            params[2] /= params[2].norm(dim=1, keepdim=True)
        theta = min_kept if theta is None else 0.99 * theta + 0.01 * min_kept
    for name, p in zip(("after_w_enc", "after_b_enc", "after_w_dec", "after_b_dec"), params):
        t["batch." + name] = p.detach().clone()
    t["batch.threshold"] = theta.reshape(1)


def jump_golden(t):
    LAMBDA, EPS, THETA0 = 0.05, 0.5, 0.2
    torch.manual_seed(9)
    w_enc = torch.randn(DIM, M) * 0.5
    b_enc = torch.randn(M) * 0.1
    w_dec = torch.randn(M, DIM) * 0.7
    b_dec = torch.randn(DIM) * 0.2
    log_t = torch.full((M,), math.log(THETA0)) + 0.3 * torch.randn(M)
    x = torch.randn(N, DIM)

    def loss_fn(we, be, wd, bd, lt):
        pre = x @ we + be
        theta = lt.exp()
        z = JumpReLUFn.apply(pre, theta, EPS)
        xh = z @ wd + bd
        mse = ((xh - x) ** 2).mean()
        l0 = StepFn.apply(pre, theta, EPS).sum(1).mean()
        return mse + LAMBDA * l0, mse, LAMBDA * l0

    params = [p.clone().requires_grad_(True) for p in (w_enc, b_enc, w_dec, b_dec, log_t)]
    total, mse, sp = loss_fn(*params)
    total.backward()
    for k, v in {"w_enc": w_enc, "b_enc": b_enc, "w_dec": w_dec, "b_dec": b_dec, "log_threshold": log_t, "x": x,
                 "loss": total.detach().reshape(1), "reconstruction": mse.detach().reshape(1), "sparsity": sp.detach().reshape(1),
                 "grad_w_enc": params[0].grad, "grad_b_enc": params[1].grad, "grad_w_dec": params[2].grad,
                 "grad_b_dec": params[3].grad, "grad_log_threshold": params[4].grad}.items():
        t["jump." + k] = v.clone()

    params = [p.detach().clone().requires_grad_(True) for p in (w_enc, b_enc, w_dec, b_dec, log_t)]
    opt = torch.optim.Adam(params, lr=1e-2)
    for _ in range(2):
        opt.zero_grad()
        total, _, _ = loss_fn(*params)
        total.backward()
        opt.step()
        with torch.no_grad():
            s = params[2].norm(dim=1)
            params[2] /= s[:, None]
            params[0] *= s[None, :]
            params[1] *= s
            params[4] += s.log()
    for name, p in zip(("after_w_enc", "after_b_enc", "after_w_dec", "after_b_dec", "after_log_threshold"), params):
        t["jump." + name] = p.detach().clone()


def main():
    t = {}
    batch_golden(t)
    jump_golden(t)
    save_file({k: v.contiguous() for k, v in t.items()}, sys.argv[1], metadata={"torch": torch.__version__})
    print("batch loss", t["batch.loss"].item(), "threshold", t["batch.threshold"].item())
    print("jump loss", t["jump.loss"].item(), "sparsity", t["jump.sparsity"].item(),
          "theta grads nonzero", int((t["jump.grad_log_threshold"] != 0).sum()))


if __name__ == "__main__":
    main()
