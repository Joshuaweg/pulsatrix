"""Writes golden outputs for DIFF-4: an A2D ("autoregressive to diffusion") Qwen3 checkpoint's float32
logits from the authors' own modeling code, to compare pulsatrix's CausalLM loaded with
bidirectional attention against (tests/a2d_test.cpp).

    docker run --rm -e HOME=/tmp -v "$PWD":/w -v ~/.cache/pulsatrix/golden:/golden -w /w pulsatrix-lrpref:latest bash -c \\
        "pip install -q --target /tmp/tf457 transformers==4.57.0 && PYTHONPATH=/tmp/tf457 \\
         python3 tools/golden/make_a2d_golden.py --download /golden/Qwen3-0.6B-diffusion-mdlm-v0.1"
    ... python3 tools/golden/make_a2d_golden.py --make-tiny tests/fixtures/hf_tiny/a2d_qwen3

The first downloads dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1 (config.json, model.safetensors and
tokenizer.json) into the directory. The second creates a tiny random A2D Qwen3 there (2 layers,
width 32, 4 query and 2 key/value heads of width 12, vocabulary 64). Both then write
a2d_golden.safetensors:
    ids.<k>     I64 [L]
    logits.<k>  F32 [L, V]
The ids are seeded random tokens with about a third replaced by the mask token (151669, `<|mask|>`,
for the real model; the vocabulary's last id for the tiny one), as a sampler would show the model.
Metadata records the revision and the versions.

The modeling code is the checkpoint's own modeling_qwen3.py, downloaded from the Hub at a pinned
revision. It needs transformers 4.57, the version the checkpoint was saved with: it reads
`Qwen3DecoderLayer.attention_type`, which transformers 5 removed. Not run by CI. Needs torch,
safetensors and huggingface_hub.
"""

import argparse
import importlib
import json
import os
import shutil
import sys
import tempfile

import torch
import transformers
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

REPO = "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1"
REVISION = "c8d24a3f4adaeef46881b450e1bf7d1005203bd7"
MASK_ID = 151669


def import_modeling_code():
    if not transformers.__version__.startswith("4.57"):
        sys.exit(f"transformers {transformers.__version__}: the authors' code needs 4.57 (see the docstring)")
    directory = tempfile.mkdtemp()
    shutil.copy(hf_hub_download(REPO, "modeling_qwen3.py", revision=REVISION), os.path.join(directory, "a2d_modeling_qwen3.py"))
    sys.path.insert(0, directory)
    return importlib.import_module("a2d_modeling_qwen3")


def make_tiny(out, modeling):
    config = modeling.A2DQwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                                     num_attention_heads=4, num_key_value_heads=2, head_dim=12, max_position_embeddings=128,
                                     tie_word_embeddings=True, rope_theta=10000.0)
    torch.manual_seed(0)
    model = modeling.A2DQwen3LMHeadModel(config)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.copy_(1.0 + 0.2 * torch.randn(p.shape))  # norms away from 1, so their weights are tested
    model.save_pretrained(out, safe_serialization=True)


def download(out):
    os.makedirs(out, exist_ok=True)
    for f in ("config.json", "model.safetensors", "tokenizer.json"):
        shutil.copy(hf_hub_download(REPO, f, revision=REVISION), os.path.join(out, f))


def write_golden(directory, modeling):
    model = modeling.A2DQwen3LMHeadModel.from_pretrained(directory, dtype=torch.float32)
    model.eval()
    with open(os.path.join(directory, "config.json")) as f:
        V = json.load(f)["vocab_size"]
    tiny = V < 1000
    mask_id = V - 1 if tiny else MASK_ID
    lengths = [7, 16, 1, 33] if tiny else [32, 1, 96]
    g = torch.Generator().manual_seed(1)
    out = {}
    for k, L in enumerate(lengths):
        ids = torch.randint(0, V - 1 if tiny else 151643, (L,), generator=g)  # ordinary tokens, then masks
        ids[torch.rand(L, generator=g) < 0.35] = mask_id
        with torch.no_grad():
            logits = model(input_ids=ids[None]).logits
        out[f"ids.{k}"] = ids.to(torch.int64)
        out[f"logits.{k}"] = logits[0].float()
    out = {k: v.contiguous() for k, v in out.items()}
    metadata = {"model": REPO, "revision": REVISION, "torch": torch.__version__, "transformers": transformers.__version__,
                "mask_token_id": str(mask_id), "generator": "pulsatrix tools/golden/make_a2d_golden.py"}
    save_file(out, os.path.join(directory, "a2d_golden.safetensors"), metadata=metadata)
    print(f"wrote {len(lengths)} sequences to {directory}/a2d_golden.safetensors")


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--download", metavar="DIR", help=f"download {REPO} into DIR, then write its golden file")
    g.add_argument("--make-tiny", metavar="DIR", help="create a tiny random A2D Qwen3 in DIR, then write its golden file")
    g.add_argument("--model", metavar="DIR", help="write the golden file for an A2D directory already on disk")
    a = p.parse_args()
    modeling = import_modeling_code()
    if a.download:
        download(a.download)
    if a.make_tiny:
        make_tiny(a.make_tiny, modeling)
    write_golden(a.download or a.make_tiny or a.model, modeling)


if __name__ == "__main__":
    main()
