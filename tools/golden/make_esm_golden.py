"""Writes golden outputs for PLM-2: an ESM-2 model's float32 logits, hidden states and attention
maps from transformers, to compare pulsatrix's EncoderLM against (tests/encoder_lm_test.cpp).

    python3 tools/golden/make_esm_golden.py --model ~/.cache/pulsatrix/golden/esm2_t6_8M_UR50D
    python3 tools/golden/make_esm_golden.py --make-tiny tests/fixtures/hf_tiny/esm

The first writes esm_golden.safetensors into the model directory (config.json, model.safetensors
and vocab.txt). The second first creates a tiny random ESM-2 there (2 layers, width 32, 4 heads,
MLP 48, token dropout, the real 33-token vocabulary), then does the same.

esm_golden.safetensors holds, for each sequence k:
    ids.<k>        I64 [L]          EsmTokenizer's ids (with <cls> and <eos>)
    logits.<k>     F32 [L, V]
    hidden.<k>.<i> F32 [L, H]       i = 0: the embeddings after token dropout; i = 1..layers: each
                                    layer's output; "final.<k>" is after emb_layer_norm_after
    attn.<k>.<i>   F32 [heads, L, L] layer i's attention probabilities (eager attention)
and a padded batch of sequences 0 and 2:
    batch.ids I64 [2, L], batch.mask I64 [2, L], batch.logits F32 [2, L, V]
plus tokenizer cases: tok.<j> I64, the ids of metadata "tok.<j>". Metadata records the sequences,
the model and the versions. Not run by CI. Needs torch, transformers and safetensors.
"""

import argparse
import json
import os
import shutil

import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoTokenizer, EsmConfig, EsmForMaskedLM

SEQUENCES = [
    "MKTAYIAKQRQISFVKSHFSRQ",
    # Human ubiquitin
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    # The start of GFP, with three residues masked
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEG<mask>ATYGKLTLKFICTTGKLPV<mask>WPTLVTTF<mask>YGVQCF",
    "MK<mask><mask>LLV",
]
TOKENIZER_CASES = ["MKTAYIAKQR", "MKT<mask>YI", "mkta", "MK TA Y", "MKJZX", "MK.-O", "<cls>MK<eos>", "", "M\tK\nT",
                   "MK<unk>T", "XBUZO", "abMKcd e"]


def make_tiny(out):
    config = EsmConfig(vocab_size=33, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=48,
                       position_embedding_type="rotary", token_dropout=True, mask_token_id=32, pad_token_id=1,
                       layer_norm_eps=1e-5, emb_layer_norm_before=False, hidden_act="gelu", max_position_embeddings=1026,
                       hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    model = EsmForMaskedLM(config)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, p in model.named_parameters():
            noise = torch.randn(p.shape, generator=g) * 0.2
            p.copy_(1.0 + noise if ("LayerNorm" in name or "layer_norm" in name) and name.endswith("weight") else noise)
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    vocab = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "fixtures", "esm", "vocab.txt")
    shutil.copy(vocab, os.path.join(out, "vocab.txt"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model")
    p.add_argument("--make-tiny")
    a = p.parse_args()
    directory = a.make_tiny or a.model
    if a.make_tiny:
        make_tiny(a.make_tiny)
    tok = AutoTokenizer.from_pretrained(directory)
    model = EsmForMaskedLM.from_pretrained(directory, attn_implementation="eager", torch_dtype=torch.float32).eval()

    captured = {}
    hooks = [model.esm.embeddings.register_forward_hook(lambda m, i, o: captured.__setitem__(0, o))]
    for i, layer in enumerate(model.esm.encoder.layer):
        hooks.append(layer.register_forward_hook(lambda m, inp, o, i=i: captured.__setitem__(i + 1, o[0] if isinstance(o, tuple) else o)))
    hooks.append(model.esm.encoder.emb_layer_norm_after.register_forward_hook(lambda m, i, o: captured.__setitem__("final", o)))

    out = {}
    with torch.no_grad():
        for k, s in enumerate(SEQUENCES):
            ids = tok(s, return_tensors="pt")["input_ids"]
            r = model(ids, output_attentions=True)
            out[f"ids.{k}"] = ids[0].to(torch.int64)
            out[f"logits.{k}"] = r.logits[0].float().contiguous()
            for i in range(len(model.esm.encoder.layer) + 1):
                out[f"hidden.{k}.{i}"] = captured[i][0].float().contiguous()
            out[f"final.{k}"] = captured["final"][0].float().contiguous()
            for i, att in enumerate(r.attentions):
                out[f"attn.{k}.{i}"] = att[0].float().contiguous()
        batch = tok([SEQUENCES[0], SEQUENCES[2]], return_tensors="pt", padding=True)
        r = model(batch["input_ids"], attention_mask=batch["attention_mask"])
        out["batch.ids"] = batch["input_ids"].to(torch.int64)
        out["batch.mask"] = batch["attention_mask"].to(torch.int64)
        out["batch.logits"] = r.logits.float().contiguous()
        for j, s in enumerate(TOKENIZER_CASES):
            out[f"tok.{j}"] = torch.tensor(tok(s)["input_ids"], dtype=torch.int64)
    for h in hooks:
        h.remove()
    metadata = {"model": os.path.basename(os.path.normpath(directory)), "torch": torch.__version__,
                "transformers": transformers.__version__, "sequences": json.dumps(SEQUENCES),
                "tokenizer_cases": json.dumps(TOKENIZER_CASES)}
    save_file(out, os.path.join(directory, "esm_golden.safetensors"), metadata=metadata)
    print(directory, len(out), "tensors")


if __name__ == "__main__":
    main()
