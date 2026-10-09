"""Writes golden values for FEAT-10's attribution graphs, computed in PyTorch the way circuit-tracer
defines them (Ameisen, Lindsey et al., "Circuit Tracing", 2025), for tests/circuit_tracing_test.cpp:

    python3 tools/golden/make_circuit_golden.py tests/fixtures/hf_tiny/llama tests/fixtures/circuit/llama_circuit.safetensors
    python3 tools/golden/make_circuit_golden.py tests/fixtures/hf_tiny/gemma3 tests/fixtures/circuit/gemma3_circuit.safetensors

The model is a tiny fixture, written out in PyTorch with Hugging Face's conventions: Llama (RMSNorm,
rotate-half RoPE, grouped-query attention, SwiGLU, tied head) or Gemma 3 (1 + w norms, sandwich
norms, QK-norm, query_pre_attn_scalar, per-layer RoPE, a sliding window, GELU-tanh, a scaled
embedding). A random transcoder over its MLPs: layer 0's 6 features write to the next 3 layers at
most (a CLT), layer 2's to 2 when there are 4 layers, the rest to their own; layer 1 has a skip.
Features read the MLP input; JumpReLU with threshold 0.05; position 0 is left out (features and
errors zeroed), as circuit-tracer does.

Attribution: every source's output vector (a_s times a decoder row, an error vector, a scaled
token embedding) is multiplied by a scalar z = 1, the rest of each MLP output is detached, and the
attention patterns and RMS scales are detached. A target's value is its encoder row dotted with
the MLP input (features) or its unembedding minus the mean unembedding dotted with the final norm's
output (logits); adjacency[t, s] = d target / d z_s, which is a_s W_dec J W_enc as in the paper.
Nodes are in circuit-tracer's order: features, errors (layer-major), tokens, logits. Needs torch
and safetensors.
"""

import json
import math
import os
import sys

import torch
from safetensors.torch import load_file, save_file

torch.set_default_dtype(torch.float64)
IDS = [1, 5, 9, 17, 3, 22, 40]
F, THETA = 6, 0.05


def main():
    model_dir, out = sys.argv[1], sys.argv[2]
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    w = {}
    for name in sorted(os.listdir(model_dir)):
        if name.endswith(".safetensors") and name.startswith("model"):
            w.update({k: v.to(torch.float64) for k, v in load_file(os.path.join(model_dir, name)).items()})
    d, L, H, Hkv = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    D, eps, V = cfg["head_dim"], cfg["rms_norm_eps"], cfg["vocab_size"]
    P = len(IDS)
    E = w["model.embed_tokens.weight"]  # (V, d), tied head

    gen = torch.Generator().manual_seed(5)
    while True:  # a transcoder whose pre-activations keep clear of the threshold
        tc = {}
        for l in range(L):
            tc[f"w_enc.{l}"] = torch.randn(F, d, generator=gen) * 0.4
            tc[f"b_enc.{l}"] = torch.randn(F, generator=gen) * 0.05
            tc[f"b_dec.{l}"] = torch.randn(d, generator=gen) * 0.02
            for k in range(span(l, L)):
                tc[f"w_dec.{l}.{k}"] = torch.randn(F, d, generator=gen) * 0.1
        tc["w_skip.1"] = torch.randn(d, d, generator=gen) * 0.05
        ok = True
        _, _, ins = forward(w, cfg, tc, None)
        for l in range(L):
            pre = ins[l] @ tc[f"w_enc.{l}"].T + tc[f"b_enc.{l}"]
            if ((pre - THETA).abs() < 2e-3).any():
                ok = False
        if ok:
            break

    # The forward pass with every source scaled by z.
    logits, _, ins = forward(w, cfg, tc, None)
    acts = []
    for l in range(L):
        pre = ins[l] @ tc[f"w_enc.{l}"].T + tc[f"b_enc.{l}"]
        a = pre * (pre > THETA)
        a[0] = 0
        acts.append(a.detach())
    features = [(l, p, f, acts[l][p, f].item()) for l in range(L) for p in range(P) for f in range(F) if acts[l][p, f] > 0]
    prob = torch.softmax(logits[-1], -1)
    top = prob.argsort(descending=True)[:10]
    logit_tokens, cum = [], 0.0
    for tok in top.tolist():
        logit_tokens.append(tok)
        cum += prob[tok].item()
        if cum >= 0.95:
            break
    nF, nE, nT, nL = len(features), L * P, P, len(logit_tokens)
    n = nF + nE + nT + nL
    z = torch.ones(n, requires_grad=True)
    _, outs, ins_z = forward(w, cfg, tc, (z, features, acts))
    u = E[logit_tokens] - E.mean(0)  # (nL, d)
    targets = [None] * n
    for i, (l, p, f, _) in enumerate(features):
        targets[i] = ins_z[l][p] @ tc[f"w_enc.{l}"][f]
    for k in range(nL):
        targets[nF + nE + nT + k] = outs[-1] @ u[k]
    adjacency = torch.zeros(n, n)
    values = torch.zeros(n)
    for i, tval in enumerate(targets):
        if tval is None:
            continue
        values[i] = tval.detach()
        (g,) = torch.autograd.grad(tval, z, retain_graph=True)
        adjacency[i] = g
    t = {
        "ids": torch.tensor(IDS, dtype=torch.int64),
        "logits": logits.detach(),
        "features": torch.tensor([[l, p, f] for l, p, f, _ in features], dtype=torch.int64),
        "activations": torch.tensor([a for *_, a in features]),
        "logit_tokens": torch.tensor(logit_tokens, dtype=torch.int64),
        "logit_probabilities": prob[logit_tokens].detach(),
        "adjacency": adjacency,
        "target_values": values,
    }
    t["spans"] = torch.tensor([span(l, L) for l in range(L)], dtype=torch.int64)
    t.update(tc)
    save_file({k: v.contiguous().to(torch.float32) if v.is_floating_point() else v.contiguous() for k, v in t.items()}, out,
              metadata={"torch": torch.__version__})
    print(f"{nF} features, {nL} logits, {n} nodes; |adjacency| max {adjacency.abs().max().item():.4f}")


def span(l, L):
    return min(3, L) if l == 0 else (2 if l == 2 and L >= 4 else 1)


def rms_frozen(x, weight, eps, offset=0.0):
    scale = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps).detach()
    return x * scale * (offset + weight)


def rope(x, base, D, factor=1.0):
    pos = torch.arange(x.shape[-2], dtype=torch.float64) / factor
    inv = 1.0 / (base ** (torch.arange(0, D, 2, dtype=torch.float64) / D))
    ang = pos[:, None] * inv[None, :]
    cos, sin = torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)
    rot = torch.cat([-x[..., D // 2:], x[..., :D // 2]], -1)
    return x * cos + rot * sin


def forward(w, cfg, tc, sources):
    """The model, frozen where circuit-tracer freezes it; sources=(z, features, acts) scales each
    source's output vector by its z. Returns logits, [final norm output], and each MLP's input."""
    gemma = cfg["model_type"].startswith("gemma")
    d, L, H, Hkv = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    D, eps = cfg["head_dim"], cfg["rms_norm_eps"]
    off = 1.0 if gemma else 0.0
    E = w["model.embed_tokens.weight"]
    P = len(IDS)
    x = E[IDS].clone() * (math.sqrt(d) if gemma else 1.0)
    nF = 0 if sources is None else len(sources[1])
    if sources is not None:
        z = sources[0]
        tok0 = nF + L * P
        x = x * z[tok0:tok0 + P, None]
    ins = []
    for l in range(L):
        pre = f"model.layers.{l}."
        h = rms_frozen(x, w[pre + "input_layernorm.weight"], eps, off)
        q = (h @ w[pre + "self_attn.q_proj.weight"].T).view(P, H, D).transpose(0, 1)
        k = (h @ w[pre + "self_attn.k_proj.weight"].T).view(P, Hkv, D).transpose(0, 1)
        v = (h @ w[pre + "self_attn.v_proj.weight"].T).view(P, Hkv, D).transpose(0, 1)
        causal = torch.triu(torch.ones(P, P, dtype=torch.bool), 1)
        if gemma:
            q = rms_frozen(q, w[pre + "self_attn.q_norm.weight"], eps, off)
            k = rms_frozen(k, w[pre + "self_attn.k_norm.weight"], eps, off)
            kind = cfg["layer_types"][l]
            rp = cfg["rope_parameters"][kind]
            factor = rp.get("factor", 1.0) if rp.get("rope_type") == "linear" else 1.0
            q, k = rope(q, rp["rope_theta"], D, factor), rope(k, rp["rope_theta"], D, factor)
            scale = cfg["query_pre_attn_scalar"] ** -0.5
            if kind == "sliding_attention":
                idx = torch.arange(P)
                causal = causal | (idx[None, :] <= idx[:, None] - cfg["sliding_window"])
        else:
            base = cfg["rope_parameters"]["rope_theta"]
            q, k = rope(q, base, D), rope(k, base, D)
            scale = 1 / math.sqrt(D)
        k, v = k.repeat_interleave(H // Hkv, 0), v.repeat_interleave(H // Hkv, 0)
        sc = (q @ k.transpose(-1, -2) * scale).masked_fill(causal, float("-inf"))
        a = torch.softmax(sc, -1).detach()
        o = (a @ v).transpose(0, 1).reshape(P, H * D) @ w[pre + "self_attn.o_proj.weight"].T
        if gemma:
            o = rms_frozen(o, w[pre + "post_attention_layernorm.weight"], eps, off)
            m_in = rms_frozen(x + o, w[pre + "pre_feedforward_layernorm.weight"], eps, off)
        else:
            m_in = rms_frozen(x + o, w[pre + "post_attention_layernorm.weight"], eps, off)
        x = x + o
        ins.append(m_in)
        gate = m_in @ w[pre + "mlp.gate_proj.weight"].T
        act = torch.nn.functional.gelu(gate, approximate="tanh") if gemma else torch.nn.functional.silu(gate)
        mlp = (act * (m_in @ w[pre + "mlp.up_proj.weight"].T)) @ w[pre + "mlp.down_proj.weight"].T
        if gemma:
            mlp = rms_frozen(mlp, w[pre + "post_feedforward_layernorm.weight"], eps, off)
        skip = m_in @ tc["w_skip.1"].T if l == 1 else torch.zeros_like(mlp)
        if sources is None:
            x = x + skip + (mlp - skip).detach()
            continue
        z, features, acts = sources
        # The sources writing here, each scaled by its z, plus the detached rest.
        written = torch.zeros(P, d)
        for i, (sl, p, f, a_s) in enumerate(features):
            key = f"w_dec.{sl}.{l - sl}"
            if l < sl or key not in tc:
                continue
            row = torch.zeros(P, d)
            row[p] = a_s * tc[key][f]
            written = written + z[i] * row
        recon = sum(acts[sl] @ tc[f"w_dec.{sl}.{l - sl}"] for sl in range(l + 1) if f"w_dec.{sl}.{l - sl}" in tc) + tc[f"b_dec.{l}"] + skip.detach()
        err = (mlp - recon).detach()
        err[0] = 0
        for p in range(P):
            row = torch.zeros(P, d)
            row[p] = err[p]
            written = written + z[nF + l * P + p] * row
        x = x + skip + (mlp - skip - written).detach() + written
    y = rms_frozen(x, w["model.norm.weight"], eps, off)
    return y @ E.T, [y[-1]], ins


if __name__ == "__main__":
    main()
