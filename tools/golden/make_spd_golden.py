"""Writes golden values for FEAT-9's parameter decomposition, written out in PyTorch after SPD
(Bushnaq, Braun and Sharkey, arXiv 2506.20790; goodfire-ai/param-decomp tag v1) and VPD
(Bushnaq et al., "Interpreting Language Model Parameters", 2026; nano_param_decomp/run.py) as
pulsatrix implements them, for tests/parameter_decomposition_test.cpp:

    python3 tools/golden/make_spd_golden.py tests/fixtures/spd/spd_golden.safetensors

Target: y = ReLU(x W1 + b1) W2 + b2, x (4, 3), hidden 4, output 2; weights (in, out) as pulsatrix
stores them, biases frozen. Each weight W ≈ V U, V (in, C = 5), U (C, out); each subcomponent's
gate z = w_out · GELU(w_in h + b_in) + b_out (hidden 3) on h = x V of the target pass.
- spd.*: masks g + (1 - g) r with g = where(z > 0, min(z, 1), 0.01 z); losses: |V U - W|² over
  the number of weights, MSE with both layers masked, MSE with one layer masked at a time
  (averaged), and 3e-3 Σ_l Σ_c mean_b u^1 with u the upper clamp. Separate fixed r per pass.
- vpd.*: the Δ weight W - V U under its own mask: y = ((x V) ⊙ (m - m_Δ)) U + m_Δ x W + b.
  Each row routes through the components of k of the 2 layers (fixed routing); unrouted rows
  and layers use masks of 1. Adversarial masks from fixed sources (coefficient 0.5). The lower
  clamp is straight-through (a leak only for gradients that raise g). Importance minimality
  2e-2 Σ_c [mean_c + 0.5 mean_c log2(1 + sum_c)] of (u + 1e-12)^0.9.
Each: the losses, every gradient, and one Adam step (lr 1e-3). Needs torch and safetensors.
"""

import sys

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

torch.set_default_dtype(torch.float64)
N, I, H, O, C, G = 4, 3, 4, 2, 5, 3
DIMS = [(I, H), (H, O)]


class LowerSte(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(0.0, 1.0)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        zero = torch.zeros_like(g)
        return torch.where(x <= 0, torch.where(g < 0, 0.01 * g, zero), torch.where(x <= 1, g, zero))


def upper(z):
    return torch.where(z > 1, 1 + 0.01 * (z - 1), F.relu(z))


def main():
    gen = torch.Generator().manual_seed(7)
    W = [torch.randn(di, do, generator=gen) for di, do in DIMS]
    b = [torch.randn(do, generator=gen) * 0.1 for _, do in DIMS]
    x = torch.rand(N, I, generator=gen)
    init = []
    for di, do in DIMS:
        init += [torch.randn(di, C, generator=gen) / di ** 0.5, torch.randn(C, do, generator=gen) / C ** 0.5,
                 torch.randn(C, G, generator=gen), torch.randn(C, G, generator=gen) * 0.1,
                 torch.randn(C, G, generator=gen) / G ** 0.5, torch.randn(C, generator=gen) * 0.1]
    noise = {
        "spd.stochastic": [torch.rand(N, C, generator=gen) for _ in DIMS],
        "spd.layerwise": [torch.rand(N, C, generator=gen) for _ in DIMS],
        "vpd.stochastic": [torch.rand(N, C, generator=gen) for _ in DIMS],
        "vpd.delta": [torch.rand(N, generator=gen) for _ in DIMS],
        "vpd.sources": [torch.rand(N, C + 1, generator=gen) for _ in DIMS],
    }
    route = torch.rand(N, 3, generator=gen)  # per row: k, then a rank per layer
    k = (1 + (route[:, 0] * 2).floor()).clamp(max=2)
    order = route[:, 1:].argsort(dim=1)
    routed = torch.zeros(2, N, dtype=torch.bool)
    for q in range(N):
        for i in range(int(k[q])):
            routed[order[q, i], q] = True

    t = {"x": x, "route": route}
    for l in range(2):
        t[f"w{l}"], t[f"b{l}"] = W[l], b[l]
        for name, v in zip(("V", "U", "gin", "gib", "gout", "gob"), init[6 * l:6 * l + 6]):
            t[f"init.{l}.{name}"] = v
    for key, vals in noise.items():
        for l, v in enumerate(vals):
            t[f"{key}.{l}"] = v

    def forward(params, masks):  # masks[l]: None (target) or (m, m_delta or None)
        a = x
        acts = []
        for l in range(2):
            acts.append(a)
            V, U = params[6 * l], params[6 * l + 1]
            if masks is None or masks[l] is None:
                z = a @ W[l]
            else:
                m, md = masks[l]
                if md is None:
                    z = ((a @ V) * m) @ U
                else:
                    z = ((a @ V) * (m - md[:, None])) @ U + md[:, None] * (a @ W[l])
            z = z + b[l]
            a = F.relu(z) if l == 0 else z
        return a, acts

    def gates(params, acts):
        out = []
        for l in range(2):
            V, _, gin, gib, gout, gob = params[6 * l:6 * l + 6]
            h = acts[l] @ V
            hid = F.gelu(h.unsqueeze(-1) * gin + gib)
            out.append((hid * gout).sum(-1) + gob)
        return out

    def losses(params, method):
        y_t, acts = forward(params, None)
        y_t = y_t.detach()
        n_w = sum(w.numel() for w in W)
        faith = sum(((params[6 * l] @ params[6 * l + 1] - W[l]) ** 2).sum() for l in range(2)) / n_w
        z = gates(params, acts)
        mse = lambda y: ((y - y_t) ** 2).mean()
        if method == "spd":
            lower = [torch.where(zl > 0, zl.clamp(max=1), 0.01 * zl) for zl in z]
            up = [upper(zl) for zl in z]
            rs, rl = noise["spd.stochastic"], noise["spd.layerwise"]
            M = [(lower[l] + (1 - lower[l]) * rs[l], None) for l in range(2)]
            stoch = mse(forward(params, M)[0])
            lw = sum(mse(forward(params, [(lower[l] + (1 - lower[l]) * rl[l], None) if j == l else None for j in range(2)])[0])
                     for l in range(2)) / 2
            imp = sum(u.sum(-1).mean() for u in up)
            total = faith + stoch + lw + 3e-3 * imp
            return total, {"faithfulness": faith, "stochastic": stoch, "layerwise": lw, "importance": imp}
        lower = [LowerSte.apply(zl) for zl in z]
        up = [upper(zl) for zl in z]
        rs, rd, src = noise["vpd.stochastic"], noise["vpd.delta"], noise["vpd.sources"]
        M = []
        for l in range(2):
            m = torch.where(routed[l][:, None], lower[l] + (1 - lower[l]) * rs[l], torch.ones(N, C))
            md = torch.where(routed[l], rd[l], torch.ones(N))
            M.append((m, md))
        stoch = mse(forward(params, M)[0])
        A = [(lower[l] + (1 - lower[l]) * src[l][:, :C], src[l][:, C]) for l in range(2)]
        adv = mse(forward(params, A)[0])
        imp = torch.zeros(())
        for u in up:
            vals = (u + 1e-12) ** 0.9
            s = vals.sum(0)
            mean = s / N
            imp = imp + (mean + 0.5 * mean * torch.log2(1 + s)).sum()
        total = faith + stoch + 0.5 * adv + 2e-2 * imp
        return total, {"faithfulness": faith, "stochastic": stoch, "adversarial": adv, "importance": imp}

    for method in ("spd", "vpd"):
        params = [p.clone().requires_grad_(True) for p in init]
        total, parts = losses(params, method)
        total.backward()
        t[f"{method}.total"] = total.detach().reshape(1)
        for key, v in parts.items():
            t[f"{method}.{key}"] = v.detach().reshape(1)
        for l in range(2):
            for j, name in enumerate(("V", "U", "gin", "gib", "gout", "gob")):
                t[f"{method}.grad.{l}.{name}"] = params[6 * l + j].grad.clone()
        params = [p.detach().clone().requires_grad_(True) for p in init]
        opt = torch.optim.Adam(params, lr=1e-3)
        opt.zero_grad()
        losses(params, method)[0].backward()
        opt.step()
        for l in range(2):
            for j, name in enumerate(("V", "U", "gin", "gib", "gout", "gob")):
                t[f"{method}.after.{l}.{name}"] = params[6 * l + j].detach().clone()
        print(method, {k: round(v.item(), 8) for k, v in parts.items()}, "total", total.item())
    save_file({k: v.contiguous().to(torch.float32) if v.is_floating_point() else v.contiguous() for k, v in t.items()}, sys.argv[1],
              metadata={"torch": torch.__version__})


if __name__ == "__main__":
    main()
