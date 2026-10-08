"""Writes golden values for FEAT-5's skip transcoder, written out in PyTorch from Dunefsky et al.
(arXiv 2406.11944) and Paulo et al. (arXiv 2501.18823) as pulsatrix implements it, for
tests/transcoder_test.cpp:

    python3 tools/golden/make_transcoder_golden.py tests/fixtures/sae/transcoder_golden.safetensors

z = ReLU(TopK_k(x W_enc + b_enc)), ŷ = z W_dec + b_dec + x W_skip, loss mean((ŷ - y)²) plus AuxK:
the k_aux largest dead latents predict e = y - ŷ (no gradient) without the bias, normalized MSE
weighted by 1/32. The decoder's weight gradient loses its component along each row. Two Adam
steps (lr 1e-2), each followed by unit decoder rows. Input size 6, output 5, so the shapes are
checked. Weights are (in, out) as pulsatrix stores them. Needs torch and safetensors.
"""

import sys

import torch
from safetensors.torch import save_file

IN, OUT, M, N, K, K_AUX, ALPHA = 6, 5, 20, 9, 3, 4, 1.0 / 32


def project(w, g):
    return g - (g * w).sum(1, keepdim=True) / (w * w).sum(1, keepdim=True) * w


def main():
    torch.manual_seed(11)
    w_enc = torch.randn(IN, M) * 0.5
    b_enc = torch.randn(M) * 0.1
    w_dec = torch.randn(M, OUT)
    w_dec = w_dec / w_dec.norm(dim=1, keepdim=True)
    b_dec = torch.randn(OUT) * 0.2
    w_skip = torch.randn(IN, OUT) * 0.3
    x = torch.randn(N, IN)
    y = torch.tanh(x @ torch.randn(IN, OUT)) + 0.1 * torch.randn(N, OUT)
    dead = torch.tensor([0, 3, 8, 12, 19], dtype=torch.int64)

    def loss_fn(we, be, wd, bd, ws):
        pre = x @ we + be
        vals, idx = pre.topk(K, dim=1)
        z = torch.zeros_like(pre).scatter(1, idx, torch.relu(vals))
        yh = z @ wd + bd + x @ ws
        mse = ((yh - y) ** 2).mean()
        e = (y - yh).detach()
        pre_dead = pre[:, dead]
        avals, aidx = pre_dead.topk(min(K_AUX, len(dead)), dim=1)
        z_aux = torch.zeros_like(pre_dead).scatter(1, aidx, torch.relu(avals))
        aux = (((z_aux @ wd[dead] - e) ** 2).sum(1) / (e ** 2).sum(1)).mean()
        return mse + ALPHA * aux, mse, ALPHA * aux

    params = [p.clone().requires_grad_(True) for p in (w_enc, b_enc, w_dec, b_dec, w_skip)]
    total, mse, aux = loss_fn(*params)
    total.backward()
    grads = [p.grad.clone() for p in params]
    grads[2] = project(params[2].detach(), grads[2])
    t = {"w_enc": w_enc, "b_enc": b_enc, "w_dec": w_dec, "b_dec": b_dec, "w_skip": w_skip, "x": x, "y": y, "dead": dead,
         "loss": total.detach().reshape(1), "reconstruction": mse.detach().reshape(1), "aux": aux.detach().reshape(1),
         "grad_w_enc": grads[0], "grad_b_enc": grads[1], "grad_w_dec": grads[2], "grad_b_dec": grads[3], "grad_w_skip": grads[4]}

    params = [p.detach().clone().requires_grad_(True) for p in (w_enc, b_enc, w_dec, b_dec, w_skip)]
    opt = torch.optim.Adam(params, lr=1e-2)
    for _ in range(2):
        opt.zero_grad()
        total, _, _ = loss_fn(*params)
        total.backward()
        params[2].grad = project(params[2].detach(), params[2].grad)
        opt.step()
        with torch.no_grad():
            params[2] /= params[2].norm(dim=1, keepdim=True)
    for name, p in zip(("after_w_enc", "after_b_enc", "after_w_dec", "after_b_dec", "after_w_skip"), params):
        t[name] = p.detach().clone()
    save_file({k: v.contiguous() for k, v in t.items()}, sys.argv[1], metadata={"torch": torch.__version__})
    print("loss", t["loss"].item(), "aux", t["aux"].item())


if __name__ == "__main__":
    main()
