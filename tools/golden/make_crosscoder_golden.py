"""Writes golden values for FEAT-8's crosscoders, written out in PyTorch after Lindsey et al.
(Transformer Circuits, 2024), Minder et al. (arXiv 2504.02922) and Kassem et al. (arXiv
2603.04426) as pulsatrix implements them, for tests/crosscoder_test.cpp:

    python3 tools/golden/make_crosscoder_golden.py tests/fixtures/sae/crosscoder_golden.safetensors

Weights are (in, out) as pulsatrix stores them; source s is columns [s d, (s+1) d) of x.
- l1.*: S = 3 sources of d = 4, m = 10. f = ReLU(x W_enc + b_enc), x̂ = f W_dec + b_dec,
  loss mean((x̂ - x)²) + λ mean_rows Σ_i f_i Σ_s |d_i^s| with λ = 0.1.
- btk.*: S = 2, d = 5, m = 12, k = 3. The batch's N k largest f_i Σ_s |d_i^s| keep f_i (the
  selection has no gradient). AuxK: the k_aux = 4 dead latents of largest scaled value per row
  reconstruct e = x - x̂ (no gradient) without the bias, normalized squared error times 1/32.
- delta.*: the Delta-Crosscoder, S = 2, d = 5, m = 15: latents 0-2 shared (k_shared = 2), 3-14
  delta (k = 2), each partition its own BatchTopK; the encoder reads x / 2; plus
  λ_Δ mean((z_Δ (W_dec^1 - W_dec^0) - (x^1 - x^0))²) with λ_Δ = 0.5, and AuxK.
Each: the loss, its gradients, and two Adam steps (lr 1e-2) without decoder normalization.
Needs torch and safetensors.
"""

import sys

import torch
from safetensors.torch import save_file


def scaled_norms(wd, S, d):
    return wd.reshape(wd.shape[0], S, d).norm(dim=2).sum(1)


def batch_topk_mask(v, k):
    n = v.shape[0]
    flat = v.flatten()
    idx = flat.topk(n * k).indices
    mask = torch.zeros_like(flat)
    mask[idx] = 1.0
    return (mask * (flat > 0)).reshape(v.shape)


def auxk(pre, wd, norms, dead, k_aux, x, xh, alpha):
    if alpha == 0 or len(dead) == 0:
        return torch.zeros(())
    e = (x - xh).detach()
    with torch.no_grad():
        v = torch.relu(pre[:, dead]) * norms[dead]
        idx = v.topk(min(k_aux, len(dead)), dim=1).indices
    sel = torch.zeros_like(pre[:, dead]).scatter(1, idx, 1.0)
    z_aux = torch.relu(pre[:, dead]) * sel
    e_hat = z_aux @ wd[dead]
    return alpha * (((e_hat - e) ** 2).sum(1) / (e ** 2).sum(1)).mean()


def make(name, cfg, t):
    S, d, m, N = cfg["S"], cfg["d"], cfg["m"], cfg["N"]
    D = S * d
    g = torch.Generator().manual_seed(cfg["seed"])
    w_enc = torch.randn(D, m, generator=g) * 0.5
    b_enc = torch.randn(m, generator=g) * 0.1
    w_dec = torch.randn(m, D, generator=g) * 0.5
    b_dec = torch.randn(D, generator=g) * 0.1
    x = torch.randn(N, D, generator=g)
    dead = torch.tensor(cfg.get("dead", []), dtype=torch.int64)

    def loss_fn(we, be, wd, bd):
        xin = x * 0.5 if cfg["kind"] == "delta" else x
        pre = xin @ we + be
        norms = scaled_norms(wd, S, d)
        if cfg["kind"] == "l1":
            f = torch.relu(pre)
            xh = f @ wd + bd
            mse = ((xh - x) ** 2).mean()
            sparsity = cfg["l1"] * (f * norms).sum(1).mean()
            return mse + sparsity, mse, sparsity, torch.zeros(())
        with torch.no_grad():
            v = torch.relu(pre) * norms
            if cfg["kind"] == "delta":
                sh = cfg["shared"]
                mask = torch.cat([batch_topk_mask(v[:, :sh], cfg["k_shared"]), batch_topk_mask(v[:, sh:], cfg["k"])], dim=1)
            else:
                mask = batch_topk_mask(v, cfg["k"])
        z = torch.relu(pre) * mask
        xh = z @ wd + bd
        mse = ((xh - x) ** 2).mean()
        aux = auxk(pre, wd, norms, dead, cfg["k_aux"], x, xh, 1.0 / 32)
        dl = torch.zeros(())
        if cfg["kind"] == "delta":
            sh = cfg["shared"]
            target = x[:, d:] - x[:, :d]
            pred = z[:, sh:] @ (wd[sh:, d:] - wd[sh:, :d])
            dl = cfg["lambda_delta"] * ((pred - target) ** 2).mean()
        return mse + aux + dl, mse, aux, dl

    params = [p.clone().requires_grad_(True) for p in (w_enc, b_enc, w_dec, b_dec)]
    total, mse, sparsity, dl = loss_fn(*params)
    total.backward()
    t.update({f"{name}.w_enc": w_enc, f"{name}.b_enc": b_enc, f"{name}.w_dec": w_dec, f"{name}.b_dec": b_dec, f"{name}.x": x,
              f"{name}.loss": total.detach().reshape(1), f"{name}.reconstruction": mse.detach().reshape(1),
              f"{name}.sparsity": sparsity.detach().reshape(1), f"{name}.delta": dl.detach().reshape(1)})
    if len(dead):
        t[f"{name}.dead"] = dead
    for n, p in zip(("w_enc", "b_enc", "w_dec", "b_dec"), params):
        t[f"{name}.grad_{n}"] = p.grad.clone()

    params = [p.detach().clone().requires_grad_(True) for p in (w_enc, b_enc, w_dec, b_dec)]
    opt = torch.optim.Adam(params, lr=1e-2)
    for _ in range(2):
        opt.zero_grad()
        total, _, _, _ = loss_fn(*params)
        total.backward()
        opt.step()
    for n, p in zip(("w_enc", "b_enc", "w_dec", "b_dec"), params):
        t[f"{name}.after_{n}"] = p.detach().clone()
    print(name, "loss", total.item())


def main():
    t = {}
    make("l1", {"kind": "l1", "S": 3, "d": 4, "m": 10, "N": 7, "l1": 0.1, "seed": 1}, t)
    make("btk", {"kind": "btk", "S": 2, "d": 5, "m": 12, "N": 9, "k": 3, "k_aux": 4, "dead": [0, 4, 7, 9, 11], "seed": 2}, t)
    make("delta", {"kind": "delta", "S": 2, "d": 5, "m": 15, "N": 9, "k": 2, "k_shared": 2, "shared": 3, "k_aux": 4,
                   "dead": [1, 5, 8, 13], "lambda_delta": 0.5, "seed": 3}, t)
    save_file({k: v.contiguous() for k, v in t.items()}, sys.argv[1], metadata={"torch": torch.__version__})


if __name__ == "__main__":
    main()
