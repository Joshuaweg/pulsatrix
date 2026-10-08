"""Writes golden values for FEAT-2: a TopK sparse autoencoder's loss and gradients, written out in
PyTorch from Gao et al. (arXiv 2406.04093) as pulsatrix implements it, to compare
TopKSparseAutoencoder against (tests/topk_sparse_autoencoder_test.cpp).

    python3 tools/golden/make_topk_sae_golden.py tests/fixtures/sae/topk_sae_golden.safetensors

The model: z = ReLU(TopK_k((x - b_dec) W_enc + b_enc)), x̂ = z W_dec + b_dec; weights are (in, out)
as pulsatrix stores them. The loss: mean((x̂ - x)²) + aux_coefficient * mean over rows of
|ê - e|² / |e|², where e = x - x̂ (no gradient) and ê = z_aux W_dec, z_aux being ReLU of the k_aux
largest pre-activations among the dead latents. The decoder's weight gradient then loses its
component along each row. Two Adam steps follow (lr 1e-2, torch's defaults otherwise), each
followed by rescaling the decoder rows to unit norm.

Stored: weights w_enc, b_enc, w_dec, b_dec; x; dead (I64 indices); loss, reconstruction, aux;
grad_w_enc, grad_b_enc, grad_w_dec, grad_b_dec; after_w_enc, after_b_enc, after_w_dec,
after_b_dec. Metadata: k, k_aux, aux_coefficient. Needs torch and safetensors.
"""

import sys

import torch
from safetensors.torch import save_file

DIM, M, N, K, K_AUX, ALPHA = 6, 20, 9, 3, 4, 1.0 / 32


def project(w, g):
    norm2 = (w * w).sum(1, keepdim=True)
    return g - (g * w).sum(1, keepdim=True) / norm2 * w


def main():
    out = sys.argv[1]
    torch.manual_seed(5)
    w_enc = torch.randn(DIM, M) * 0.5
    b_enc = torch.randn(M) * 0.1
    w_dec = torch.randn(M, DIM)
    w_dec = w_dec / w_dec.norm(dim=1, keepdim=True)
    b_dec = torch.randn(DIM) * 0.2
    x = torch.randn(N, DIM)
    dead = torch.tensor([1, 4, 7, 11, 13, 17], dtype=torch.int64)
    params = [p.clone().requires_grad_(True) for p in (w_enc, b_enc, w_dec, b_dec)]

    def loss_fn(we, be, wd, bd):
        pre = (x - bd) @ we + be
        vals, idx = pre.topk(K, dim=1)
        z = torch.zeros_like(pre).scatter(1, idx, torch.relu(vals))
        xh = z @ wd + bd
        mse = ((xh - x) ** 2).mean()
        e = (x - xh).detach()
        pre_dead = pre[:, dead]
        avals, aidx = pre_dead.topk(min(K_AUX, len(dead)), dim=1)
        z_aux = torch.zeros_like(pre_dead).scatter(1, aidx, torch.relu(avals))
        e_hat = z_aux @ wd[dead]
        aux = (((e_hat - e) ** 2).sum(1) / (e ** 2).sum(1)).mean()
        return mse + ALPHA * aux, mse, aux

    total, mse, aux = loss_fn(*params)
    total.backward()
    grads = [p.grad.clone() for p in params]
    grads[2] = project(params[2].detach(), grads[2])
    tensors = {"w_enc": w_enc, "b_enc": b_enc, "w_dec": w_dec, "b_dec": b_dec, "x": x, "dead": dead,
               "loss": total.detach().reshape(1), "reconstruction": mse.detach().reshape(1), "aux": (ALPHA * aux).detach().reshape(1),
               "grad_w_enc": grads[0], "grad_b_enc": grads[1], "grad_w_dec": grads[2], "grad_b_dec": grads[3]}

    params = [p.detach().clone().requires_grad_(True) for p in (w_enc, b_enc, w_dec, b_dec)]
    opt = torch.optim.Adam(params, lr=1e-2)
    for _ in range(2):
        opt.zero_grad()
        total, _, _ = loss_fn(*params)
        total.backward()
        params[2].grad = project(params[2].detach(), params[2].grad)
        opt.step()
        with torch.no_grad():
            params[2] /= params[2].norm(dim=1, keepdim=True)
    for name, p in zip(("after_w_enc", "after_b_enc", "after_w_dec", "after_b_dec"), params):
        tensors[name] = p.detach().clone()
    save_file({k: v.contiguous() for k, v in tensors.items()}, out,
              metadata={"k": str(K), "k_aux": str(K_AUX), "aux_coefficient": str(ALPHA), "torch": torch.__version__})
    print("loss", tensors["loss"].item(), "aux", tensors["aux"].item())


if __name__ == "__main__":
    main()
