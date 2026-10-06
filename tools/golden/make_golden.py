"""Writes golden logits for LLM-6: a Hugging Face model's float32 logits on a fixed text set, to
compare pulsatrix against (pulsatrix_golden, tests/golden_logits_test.cpp).

    python3 tools/golden/make_golden.py --model HuggingFaceTB/SmolLM2-135M --out golden/SmolLM2-135M

downloads the model (config.json and safetensors only) into the output directory, then writes
golden.safetensors beside it:
    ids.<k>     I64 [L]       the tokens of text k (the tokenizer runs here, in Python)
    logits.<k>  F32 [L, V]    transformers' logits for them, eager attention, float32
and metadata recording the model, its resolved Hub revision, and the versions used. Texts come
from tests/fixtures/golden/texts.txt, one per line ("\\n" escapes a newline), cut to --max-tokens.

For a model directory without a tokenizer (the tiny test models), --random-ids N writes N
sequences of seeded random ids instead.

Not run by CI. Needs torch, transformers, safetensors and huggingface_hub.
"""

import argparse
import os

import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

TEXTS = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "fixtures", "golden", "texts.txt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="a Hub id, or a local directory")
    p.add_argument("--revision", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--texts", default=TEXTS)
    p.add_argument("--max-tokens", type=int, default=48)
    p.add_argument("--random-ids", type=int, default=0)
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    revision = "local"
    path = a.model
    if not os.path.isdir(a.model):
        from huggingface_hub import HfApi, snapshot_download
        revision = HfApi().model_info(a.model, revision=a.revision).sha
        path = snapshot_download(a.model, revision=revision, local_dir=a.out,
                                 allow_patterns=["config.json", "*.safetensors", "*.safetensors.index.json",
                                                 "tokenizer*", "generation_config.json"])
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation="eager").eval()

    if a.random_ids:
        g = torch.Generator().manual_seed(0)
        lengths = [5, 9, 16, 31][: a.random_ids] + [12] * max(0, a.random_ids - 4)
        sequences = [torch.randint(0, model.config.vocab_size, (n,), generator=g).tolist() for n in lengths]
    else:
        tok = AutoTokenizer.from_pretrained(path)
        with open(a.texts, encoding="utf-8") as f:
            lines = [line.rstrip("\n").replace("\\n", "\n") for line in f if line.strip()]
        sequences = [tok(line).input_ids[: a.max_tokens] for line in lines]

    tensors = {}
    with torch.no_grad():
        for k, ids in enumerate(sequences):
            tensors["ids.%d" % k] = torch.tensor(ids, dtype=torch.int64)
            tensors["logits.%d" % k] = model(torch.tensor([ids])).logits[0].float().contiguous()
    metadata = {"model": a.model, "revision": revision, "torch": torch.__version__,
                "transformers": transformers.__version__, "dtype": "float32", "texts": str(len(sequences))}
    save_file(tensors, os.path.join(a.out, "golden.safetensors"), metadata=metadata)
    print("wrote %d sequences for %s @ %s" % (len(sequences), a.model, revision))


if __name__ == "__main__":
    main()
