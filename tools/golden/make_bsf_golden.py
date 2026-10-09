"""Writes golden values for FEAT-6's block-sparse featurizers, written out in PyTorch after Fel et
al. (arXiv 2606.25234) and their reference code (github.com/goodfire-ai/block-sparse-featurizer)
as pulsatrix implements them, for tests/block_sparse_featurizer_test.cpp:

    python3 tools/golden/make_bsf_golden.py tests/fixtures/sae/bsf_golden.safetensors

d = 6, G = 5 blocks of b = 2, N = 9 inputs; weights (in, out) as pulsatrix stores them; the loss
is mean((x̂ - x)²) [+ λ L0].
- vanilla.*: z = Π_2(x W_enc + b_enc) by block norm, x̂ = z D. Adam steps renormalize each atom.
- grass.*: z_g = Π_2(γ_g x D_gᵀ), tied, frames orthonormal; Adam steps re-orthonormalize each
  block by QR (positive diagonal), as Appendix D of the paper does.
- lasso.*: a = x W_enc + b_enc, gate = H(‖a_g‖ - θ_g), θ = softplus(10 raw), z = a·gate, the
  reference code's BlockJumpReLU straight-through estimator (Gaussian kernel), λ L0 with λ = 0.01,
  cold start of θ and the bandwidth on the first batch (target L0 2), and dual ascent on log λ.
Needs torch and safetensors.
"""

import math
import sys

import torch
from safetensors.torch import save_file

D, G, B, N, K = 6, 5, 2, 9, 2


def block_topk(z):
    zb = z.reshape(N, G, B)
    idx = zb.norm(dim=-1).topk(K, dim=1).indices
    mask = torch.zeros(N, G).scatter(1, idx, 1.0)
    return (zb * mask[:, :, None]).reshape(N, G * B)


def unit_rows(w):
    return w / w.norm(dim=1, keepdim=True).clamp_min(1e-8)


def orthonormal_blocks(w):
    out = []
    for g in range(G):
        q, r = torch.linalg.qr(w[g * B:(g + 1) * B].t())
        q = q * torch.sign(torch.diagonal(r))[None, :]
        out.append(q.t())
    return torch.cat(out)


def record(t, prefix, names, params, loss_parts):
    for name, p in zip(names, params):
        t[f"{prefix}.grad_{name}"] = p.grad.clone()
    for k, v in loss_parts.items():
        t[f"{prefix}.{k}"] = v.detach().reshape(1)


def adam_twice(params, loss_fn, project):
    opt = torch.optim.Adam(params, lr=1e-2)
    for _ in range(2):
        opt.zero_grad()
        loss_fn(*params)[0].backward()
        opt.step()
        with torch.no_grad():
            project(params)
    return params


def vanilla(t, x):
    torch.manual_seed(1)
    w_dec = torch.randn(G * B, D) * 0.8
    w_enc = torch.randn(D, G * B) * 0.6
    b_enc = torch.randn(G * B) * 0.1

    def loss_fn(we, be, wd):
        mse = ((block_topk(x @ we + be) @ wd - x) ** 2).mean()
        return mse, {"loss": mse}

    params = [p.clone().requires_grad_(True) for p in (w_enc, b_enc, w_dec)]
    total, parts = loss_fn(*params)
    total.backward()
    t.update({"vanilla.w_enc": w_enc, "vanilla.b_enc": b_enc, "vanilla.w_dec": w_dec})
    record(t, "vanilla", ("w_enc", "b_enc", "w_dec"), params, parts)

    def project(ps):
        ps[2].copy_(unit_rows(ps[2]))

    after = adam_twice([p.detach().clone().requires_grad_(True) for p in (w_enc, b_enc, w_dec)], loss_fn, project)
    for name, p in zip(("w_enc", "b_enc", "w_dec"), after):
        t[f"vanilla.after_{name}"] = p.detach().clone()


def grassmannian(t, x):
    torch.manual_seed(2)
    frames = orthonormal_blocks(torch.randn(G * B, D))
    gamma = 1.0 + 0.3 * torch.randn(G)

    def loss_fn(fr, gm):
        z = block_topk((x @ fr.t()) * gm.repeat_interleave(B)[None, :])
        mse = ((z @ fr - x) ** 2).mean()
        return mse, {"loss": mse}

    params = [p.clone().requires_grad_(True) for p in (frames, gamma)]
    total, parts = loss_fn(*params)
    total.backward()
    t.update({"grass.frames": frames, "grass.gamma": gamma})
    record(t, "grass", ("frames", "gamma"), params, parts)

    def project(ps):
        ps[0].copy_(orthonormal_blocks(ps[0]))

    after = adam_twice([p.detach().clone().requires_grad_(True) for p in (frames, gamma)], loss_fn, project)
    for name, p in zip(("frames", "gamma"), after):
        t[f"grass.after_{name}"] = p.detach().clone()


class BlockJumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gn, theta, bandwidth):
        ctx.save_for_backward(gn, theta)
        ctx.bw = float(bandwidth)
        return (gn > theta).to(gn.dtype)

    @staticmethod
    def backward(ctx, g):
        gn, theta = ctx.saved_tensors
        h = ctx.bw / 2
        k = torch.exp(-0.5 * ((gn - theta) / h) ** 2) / (ctx.bw * (2.0 * math.pi) ** 0.5)
        gk = g * k
        return gk, -gk.sum(0), None


def group_lasso(t, x):
    GAIN, MULT, TARGET, COEF, DUAL = 10.0, 0.25, 2.0, 1e-2, 3e-2
    torch.manual_seed(3)
    w_dec = unit_rows(torch.randn(G * B, D))
    w_enc = w_dec.t().clone() + 0.2 * torch.randn(D, G * B)
    b_enc = torch.randn(G * B) * 0.1
    raw0 = torch.zeros(G)

    class State:
        def __init__(self):
            self.inited, self.bw, self.log_lambda = False, 1.0, math.log(COEF)

    def make_loss(state):
        def loss_fn(we, be, wd, raw):
            a = (x @ we + be).reshape(N, G, B)
            gn = a.norm(dim=-1)
            if not state.inited:
                thr = torch.quantile(gn.flatten().float(), 1.0 - TARGET / G).clamp_min(1e-3)
                with torch.no_grad():
                    raw.copy_(torch.full((G,), float(torch.log(torch.expm1(thr)) / GAIN)))
                state.bw = float(gn.detach().std().clamp_min(1e-6) * MULT)
                state.inited = True
            state.bw = 0.99 * state.bw + 0.01 * float(gn.detach().std().clamp_min(1e-6) * MULT)
            theta = torch.nn.functional.softplus(GAIN * raw)
            gate = BlockJumpReLU.apply(gn, theta, state.bw)
            z = (a * gate[:, :, None]).reshape(N, G * B)
            mse = ((z @ wd - x) ** 2).mean()
            l0 = gate.sum(-1).mean()
            lam = math.exp(state.log_lambda)
            err = (float(l0.detach()) - TARGET) / TARGET
            state.log_lambda += DUAL * max(-1.0, min(1.0, err))
            state.log_lambda = min(max(state.log_lambda, math.log(1e-8)), math.log(1e2))
            return mse + lam * l0, {"loss": mse + lam * l0, "reconstruction": mse, "sparsity": lam * l0}
        return loss_fn

    state = State()
    params = [p.clone().requires_grad_(True) for p in (w_enc, b_enc, w_dec, raw0)]
    total, parts = make_loss(state)(*params)
    total.backward()
    t.update({"lasso.w_enc": w_enc, "lasso.b_enc": b_enc, "lasso.w_dec": w_dec})
    record(t, "lasso", ("w_enc", "b_enc", "w_dec", "threshold_raw"), params, parts)
    t["lasso.log_lambda_after_one"] = torch.tensor([state.log_lambda])

    def project(ps):
        ps[2].copy_(unit_rows(ps[2]))

    state = State()
    after = adam_twice([p.detach().clone().requires_grad_(True) for p in (w_enc, b_enc, w_dec, raw0)], make_loss(state), project)
    for name, p in zip(("w_enc", "b_enc", "w_dec", "threshold_raw"), after):
        t[f"lasso.after_{name}"] = p.detach().clone()
    t["lasso.after_log_lambda"] = torch.tensor([state.log_lambda])
    t["lasso.after_bandwidth"] = torch.tensor([state.bw])


def main():
    torch.manual_seed(0)
    x = torch.randn(N, D)
    t = {"x": x}
    vanilla(t, x)
    grassmannian(t, x)
    group_lasso(t, x)
    save_file({k: v.contiguous().float() for k, v in t.items()}, sys.argv[1], metadata={"torch": torch.__version__})
    print({k: float(v) for k, v in t.items() if v.numel() == 1})


if __name__ == "__main__":
    main()
