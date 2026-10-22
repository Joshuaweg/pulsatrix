"""Writes golden trajectories for DIFF-4: DPLM's own reparameterized decoding (its remasking sampler)
run on an ESM-architecture checkpoint, to compare pulsatrix's UnmaskingSchedule::Remasking against
(tests/dplm_test.cpp).

    docker run --rm -v "$PWD":/w -v ~/.cache/pulsatrix/golden:/golden -w /w pulsatrix-lrpref:latest \\
        python3 tools/golden/make_dplm_golden.py --download airkingbd/dplm_150m /golden/dplm_150m
    docker run ... python3 tools/golden/make_dplm_golden.py --model tests/fixtures/hf_tiny/esm

The first downloads a DPLM checkpoint (config.json, vocab.txt and pytorch_model.bin) at a pinned
revision into the directory and converts the pickle to model.safetensors with
tools/convert/pickle_to_safetensors.py. Both then write dplm_golden.safetensors:
    input.<k>   I64 [L]      the sequence to complete: kept positions, then <mask> where DPLM fills
    steps.<k>   I64 [T, L]   the sequence the network saw at each of the T steps
    output.<k>  I64 [L]      DPLM's result
with the step count in the metadata (`steps`). Each case keeps a protein's first and last thirds and
masks the middle (DPLM's `partial_masks` marks the kept positions), and one masks every residue.

The decoding is DPLM's own code: `generate`, `forward_decoder`, `_reparam_decoding`,
`get_non_special_symbol_mask`, `initialize_output_tokens` and `forward_encoder` from
src/byprot/models/dplm/dplm.py, and `topk_masking` and `sample_from_categorical` from
src/byprot/models/utils.py, taken from the source at a pinned commit and run unchanged on a small
object holding the network and the special token ids. The repository's own imports (OmegaConf,
byprot's registry, DPLM's network classes) are not needed and not installed. Sampling is
`vanilla` at temperature 0, DPLM's deterministic path: the most likely token, scored by its
log-probability once DPLM has set the special tokens' logits to -inf. The network is
transformers' EsmForMaskedLM, which DPLM's EsmForDPLM subclasses without changing the forward pass
(it only fixes the attention mask to `input_ids != pad`, and these inputs have no padding). Not run
by CI. Needs torch, transformers, safetensors and huggingface_hub.
"""

import argparse
import ast
import math
import os
import shutil
import subprocess
import sys
import urllib.request

import numpy as np
import torch
import transformers
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file
from transformers import EsmForMaskedLM

DPLM_COMMIT = "8a2e15e53416b4536f03f79ad1f6f6a9cbd5e19d"  # bytedance/dplm, Apache-2.0
REVISIONS = {"airkingbd/dplm_150m": "49b7125a5d28c6418fcc2f3c4fe799352ac1488b",
             "airkingbd/dplm_650m": "7a7e651baa667d094aba05e9dc1cf52a3332110a"}
METHODS = ("generate", "forward_decoder", "_reparam_decoding", "get_non_special_symbol_mask", "initialize_output_tokens",
           "forward_encoder")
FUNCTIONS = ("topk_masking", "sample_from_categorical")
# Human ubiquitin, and the start of GFP.
PROTEINS = ["MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
            "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQE"]


def dplm_source(path):
    url = f"https://raw.githubusercontent.com/bytedance/dplm/{DPLM_COMMIT}/{path}"
    with urllib.request.urlopen(url) as r:
        return r.read().decode()


def extract(source, names):
    """The named functions and methods, anywhere in the file, as source text."""
    found = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            found[node.name] = ast.get_source_segment(source, node)
    missing = set(names) - set(found)
    if missing:
        sys.exit(f"not found in DPLM's source: {sorted(missing)}")
    return found


def dplm_decoder(net, vocab):
    """An object with DPLM's decoding methods bound to it, and the attributes they read."""
    namespace = {"torch": torch, "np": np, "math": math, "tqdm": lambda iterable, **kwargs: iterable}
    for name, code in extract(dplm_source("src/byprot/models/utils.py"), FUNCTIONS).items():
        exec(code, namespace)  # noqa: S102 (DPLM's own code, pinned)
    methods = {}
    for name, code in extract(dplm_source("src/byprot/models/dplm/dplm.py"), METHODS).items():
        exec(code, namespace)  # noqa: S102
        methods[name] = namespace[name]
    decoder = type("DplmDecoder", (), methods)()
    decoder.net = net
    decoder.mask_id, decoder.pad_id, decoder.bos_id = vocab["<mask>"], vocab["<pad>"], vocab["<cls>"]
    decoder.eos_id, decoder.x_id = vocab["<eos>"], vocab["X"]
    return decoder


class RecordingNet(torch.nn.Module):
    """EsmForMaskedLM as DPLM's net: logits for input_ids, and a record of every input it saw."""

    def __init__(self, model, pad_id):
        super().__init__()
        self.model, self.pad_id, self.seen = model, pad_id, []

    def forward(self, input_ids):
        self.seen.append(input_ids.clone())
        out = self.model(input_ids=input_ids, attention_mask=input_ids.ne(self.pad_id), output_hidden_states=True)
        return {"logits": out.logits, "last_hidden_state": out.hidden_states[-1]}


def download(repo, out):
    os.makedirs(out, exist_ok=True)
    revision = REVISIONS[repo]
    for f in ("config.json", "vocab.txt"):
        shutil.copy(hf_hub_download(repo, f, revision=revision), os.path.join(out, f))
    pickle = hf_hub_download(repo, "pytorch_model.bin", revision=revision)
    converter = os.path.join(os.path.dirname(__file__), "..", "convert", "pickle_to_safetensors.py")
    subprocess.run([sys.executable, converter, pickle, os.path.join(out, "model.safetensors"), "--force"], check=True)


def write_golden(directory):
    with open(os.path.join(directory, "vocab.txt")) as f:
        vocab = {line.strip(): i for i, line in enumerate(f)}
    model = EsmForMaskedLM.from_pretrained(directory, dtype=torch.float32).eval()
    net = RecordingNet(model, vocab["<pad>"])
    decoder = dplm_decoder(net, vocab)
    tiny = model.config.num_hidden_layers <= 2
    steps = 6 if tiny else 32
    cases = []
    for protein in PROTEINS:
        protein = protein[:24] if tiny else protein
        ids = [vocab["<cls>"]] + [vocab[a] for a in protein] + [vocab["<eos>"]]
        keep = torch.ones(len(ids), dtype=torch.bool)
        third = len(protein) // 3
        keep[1 + third:1 + len(protein) - third] = False
        cases.append((torch.tensor(ids), keep))
    everything = cases[0][0].clone()
    cases.append((everything, torch.zeros(len(everything), dtype=torch.bool)))

    out = {}
    for k, (ids, keep) in enumerate(cases):
        net.seen = []
        with torch.no_grad():
            result = decoder.generate(ids[None], max_iter=steps, temperature=0.0, partial_masks=keep[None],
                                      sampling_strategy="vanilla")
        start = decoder.initialize_output_tokens(ids[None], partial_masks=keep[None])[0][0]
        assert len(net.seen) == steps
        out[f"input.{k}"] = start.to(torch.int64)
        out[f"steps.{k}"] = torch.cat(net.seen).to(torch.int64)
        out[f"output.{k}"] = result[0].to(torch.int64)
    out = {k: v.contiguous() for k, v in out.items()}
    metadata = {"dplm_commit": DPLM_COMMIT, "steps": str(steps), "torch": torch.__version__,
                "transformers": transformers.__version__, "generator": "pulsatrix tools/golden/make_dplm_golden.py"}
    save_file(out, os.path.join(directory, "dplm_golden.safetensors"), metadata=metadata)
    print(f"wrote {len(cases)} trajectories of {steps} steps to {directory}/dplm_golden.safetensors")


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--download", nargs=2, metavar=("REPO", "DIR"), help=f"download one of {sorted(REVISIONS)} into DIR")
    g.add_argument("--model", metavar="DIR", help="write the golden file for an ESM-architecture directory already on disk")
    a = p.parse_args()
    if a.download:
        download(*a.download)
    write_golden(a.download[1] if a.download else a.model)


if __name__ == "__main__":
    main()
