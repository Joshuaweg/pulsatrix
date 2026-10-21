"""Writes golden outputs for DIFF-2: an MDLM (masked diffusion language model) checkpoint's float32
logits and hidden states from the authors' own modeling code, to compare pulsatrix's EncoderLM
against (tests/mdlm_test.cpp).

    docker run --rm -v "$PWD":/w -v ~/.cache/pulsatrix/golden:/golden -w /w pulsatrix-lrpref:latest \
        python3 tools/golden/make_mdlm_golden.py --download /golden/mdlm-owt
    docker run ... python3 tools/golden/make_mdlm_golden.py --make-tiny tests/fixtures/hf_tiny/mdlm

The first downloads kuleshov-group/mdlm-owt (config.json and model.safetensors) into the
directory. The second creates a tiny random MDLM there (2 blocks, width 32, 4 heads, conditioning
width 16, vocabulary 40) whose adaLN and timestep-embedding weights are random rather than zero, so
folding the constant modulation into the weights is actually tested. Both then write
mdlm_golden.safetensors:
    ids.<k>        I64 [L]
    logits.<k>     F32 [L, V]   the backbone's raw logits, before MDLM's SUBS parameterization
    hidden.<k>.<i> F32 [L, H]   i = 0: the embeddings; i = 1..blocks: each block's output
    final.<k>      F32 [L, H]   the final layer's modulated LayerNorm, before its linear layer
and a batch of sequences 0 and 1 cut to the same length: batch.ids I64 [2, L], batch.logits F32
[2, L, V]. The ids are seeded random tokens with about a third replaced by the mask token (the
vocabulary's last id), as a sampler would show the model. Metadata records the revision and the
versions.

The modeling code is the checkpoint's own modeling_mdlm.py and configuration_mdlm.py, downloaded
from the Hub at a pinned revision. It imports flash_attn, which needs CUDA, for two functions:
the rotate-half rotary embedding and packed variable-length attention. CPU stand-ins for exactly
those two are registered under flash_attn's module names instead. Not run by CI. Needs torch,
safetensors, einops and huggingface_hub.
"""

import argparse
import importlib
import importlib.machinery
import json
import os
import shutil
import sys
import tempfile
import types

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

REPO = "kuleshov-group/mdlm-owt"
REVISION = "d0958fa851335ece6c15260ce0025f030673c0fb"


def apply_rotary_emb_qkv_(qkv, cos, sin, cos_k=None, sin_k=None, interleaved=False, seqlen_offsets=0):
    """flash_attn.layers.rotary.apply_rotary_emb_qkv_ for interleaved=False: rotates q and k
    (rotate-half) in place. qkv is (batch, seqlen, 3, heads, dim); cos and sin are
    (seqlen, rotary_dim / 2)."""
    assert not interleaved and cos_k is None and sin_k is None and seqlen_offsets == 0
    rot = cos.shape[-1] * 2
    c = torch.cat([cos, cos], dim=-1)[None, :, None, :]
    s = torch.cat([sin, sin], dim=-1)[None, :, None, :]
    for i in (0, 1):
        x = qkv[:, :, i, :, :rot]
        x1, x2 = x[..., : rot // 2], x[..., rot // 2:]
        qkv[:, :, i, :, :rot] = x * c + torch.cat([-x2, x1], dim=-1) * s
    return qkv


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
    """flash_attn's packed variable-length attention, sequence by sequence. qkv is
    (total_tokens, 3, heads, dim); returns (total_tokens, heads, dim)."""
    assert dropout_p == 0.0
    out = torch.empty_like(qkv[:, 0])
    bounds = cu_seqlens.tolist()
    for b in range(len(bounds) - 1):
        q, k, v = (qkv[bounds[b]:bounds[b + 1], i].transpose(0, 1) for i in range(3))  # (heads, L, dim)
        out[bounds[b]:bounds[b + 1]] = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=softmax_scale).transpose(0, 1)
    return out


def install_flash_attn_stand_in():
    root = types.ModuleType("flash_attn")
    layers = types.ModuleType("flash_attn.layers")
    rotary = types.ModuleType("flash_attn.layers.rotary")
    interface = types.ModuleType("flash_attn.flash_attn_interface")
    rotary.apply_rotary_emb_qkv_ = apply_rotary_emb_qkv_
    interface.flash_attn_varlen_qkvpacked_func = flash_attn_varlen_qkvpacked_func
    root.layers, layers.rotary, root.flash_attn_interface = layers, rotary, interface
    for m in (root, layers, rotary, interface):
        # transformers probes installed packages with find_spec, which needs a spec
        m.__spec__ = importlib.machinery.ModuleSpec(m.__name__, None)
    sys.modules.update({"flash_attn": root, "flash_attn.layers": layers, "flash_attn.layers.rotary": rotary,
                        "flash_attn.flash_attn_interface": interface})


def import_modeling_code():
    """The checkpoint's modeling code as a package (it uses a relative import)."""
    install_flash_attn_stand_in()
    pkg = os.path.join(tempfile.mkdtemp(), "mdlm_ref")
    os.makedirs(pkg)
    open(os.path.join(pkg, "__init__.py"), "w").close()
    for f in ("configuration_mdlm.py", "modeling_mdlm.py"):
        shutil.copy(hf_hub_download(REPO, f, revision=REVISION), os.path.join(pkg, f))
    sys.path.insert(0, os.path.dirname(pkg))
    return importlib.import_module("mdlm_ref.configuration_mdlm"), importlib.import_module("mdlm_ref.modeling_mdlm")


def make_tiny(out, configuration, modeling):
    config = {"architectures": ["MDLM"], "model_type": "mdlm", "vocab_size": 40, "hidden_dim": 32, "cond_dim": 16,
              "n_blocks": 2, "n_heads": 4, "model_length": 64, "dropout": 0.0, "time_conditioning": False}
    model = modeling.MDLM(configuration.MDLMConfig(**{k: v for k, v in config.items() if k not in ("architectures", "model_type")}))
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, p in model.named_parameters():
            noise = torch.randn(p.shape, generator=g) * 0.2
            p.copy_(1.0 + noise if "norm" in name and name.endswith("weight") else noise)
    os.makedirs(out, exist_ok=True)
    tensors = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(tensors, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def download(out):
    os.makedirs(out, exist_ok=True)
    for f in ("config.json", "model.safetensors"):
        shutil.copy(hf_hub_download(REPO, f, revision=REVISION), os.path.join(out, f))


def write_golden(directory, configuration, modeling):
    with open(os.path.join(directory, "config.json")) as f:
        config = json.load(f)
    kwargs = {k: config[k] for k in ("vocab_size", "model_length", "hidden_dim", "cond_dim", "n_blocks", "n_heads", "dropout",
                                     "time_conditioning")}
    model = modeling.MDLM(configuration.MDLMConfig(**kwargs))
    model.load_state_dict(load_file(os.path.join(directory, "model.safetensors")), strict=True)
    model.eval()
    backbone = model.backbone

    V, mask_id = config["vocab_size"], config["vocab_size"] - 1
    lengths = [7, 16, 1, 33] if V < 1000 else [32, 64, 1, 128]
    g = torch.Generator().manual_seed(1)
    sequences = []
    for L in lengths:
        ids = torch.randint(0, V - 1, (L,), generator=g)
        ids[torch.rand(L, generator=g) < 0.35] = mask_id
        sequences.append(ids)

    final = {}
    backbone.output_layer.norm_final.register_forward_hook(lambda m, i, o: final.update(norm=o))

    def run(ids):
        with torch.no_grad():
            logits, hidden = model(input_ids=ids, timesteps=torch.zeros(ids.shape[0]), output_hidden_states=True, return_dict=False)
            c = F.silu(backbone.sigma_map(torch.zeros(ids.shape[0])))
            shift, scale = backbone.output_layer.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        return logits, hidden, final["norm"] * (1 + scale) + shift

    out = {}
    for k, ids in enumerate(sequences):
        logits, hidden, modulated = run(ids[None])
        out[f"ids.{k}"] = ids.to(torch.int64)
        out[f"logits.{k}"] = logits[0].float()
        for i, h in enumerate(hidden):
            out[f"hidden.{k}.{i}"] = h[0].float()
        out[f"final.{k}"] = modulated[0].float()
    L = min(lengths[0], lengths[1])
    batch = torch.stack([sequences[0][:L], sequences[1][:L]])
    out["batch.ids"] = batch.to(torch.int64)
    out["batch.logits"] = run(batch)[0].float()
    out = {k: v.contiguous() for k, v in out.items()}
    metadata = {"model": REPO, "revision": REVISION, "torch": torch.__version__,
                "generator": "pulsatrix tools/golden/make_mdlm_golden.py"}
    save_file(out, os.path.join(directory, "mdlm_golden.safetensors"), metadata=metadata)
    print(f"wrote {len(sequences)} sequences and a batch to {directory}/mdlm_golden.safetensors")


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--download", metavar="DIR", help="download kuleshov-group/mdlm-owt into DIR, then write its golden file")
    g.add_argument("--make-tiny", metavar="DIR", help="create a tiny random MDLM in DIR, then write its golden file")
    g.add_argument("--model", metavar="DIR", help="write the golden file for an MDLM directory already on disk")
    a = p.parse_args()
    configuration, modeling = import_modeling_code()
    if a.download:
        download(a.download)
    if a.make_tiny:
        make_tiny(a.make_tiny, configuration, modeling)
    write_golden(a.download or a.make_tiny or a.model, configuration, modeling)


if __name__ == "__main__":
    main()
