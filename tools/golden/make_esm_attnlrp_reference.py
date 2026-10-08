"""Writes AttnLRP reference relevance for PLM-6: LXT's AttnLRP rules on transformers' ESM-2, to
compare pulsatrix's EncoderExplainer against (tests/protein_explanations_test.cpp).

    docker run --rm -v "$PWD":/w -v ~/.cache/pulsatrix/golden:/golden -w /w pulsatrix-lrpref:latest \
        python3 tools/golden/make_esm_attnlrp_reference.py /golden/esm2_t6_8M_UR50D
    docker run ... python3 tools/golden/make_esm_attnlrp_reference.py tests/fixtures/hf_tiny/esm

LXT 2.1 has no ESM patch, so this applies its rules the way its BERT patch does: the identity
rule on LayerNorm (lxt's layer_norm_forward, which stops the gradient through the standard
deviation) and on both GELUs (identity_rule_implicit), LXT's attention wrapper on the eager
attention function (the uniform rule at both matmuls), and plain gradient * input elsewhere,
which is the epsilon rule with epsilon -> 0. Relevance at the output is the explained value itself
(a unit gradient). --plain-gradient skips the rules: a negative control.

esm_attnlrp.safetensors (or esm_plain_gradient.safetensors), for each sequence k:
    ids.<k>                 I64 [T]       <cls> + residues + <eos>, unmasked
    masked_ids.<k>          I64 [T]       the same with residue token p masked
    target.<k>              I64 [4]       p (token index), the explained token (the likeliest amino
                                          acid at p), the wild type's token, the residue head's
                                          token index
    mask_value.<k>          F32 [1]       the explained logit
    mask_relevance.<k>      F32 [T]       relevance per token (embedding * gradient, summed)
    mask_layers.<k>         F32 [layers+1, T]  relevance at the embeddings and each layer's output
    mutation_relevance.<k>  F32 [T]       explaining logit(token) - logit(wild type) at p, masked
    protein_head_relevance.<k> F32 [T]    explaining head_weight . mean(last hidden over residues) + bias, unmasked
    residue_head_relevance.<k> F32 [T]    explaining head_weight . last hidden at the residue head's token + bias
and head_weight F32 [hidden], head_bias F32 [1]. Metadata: the sequences, model and versions.
"""

import argparse
import json
import os

import torch
import transformers
import transformers.pytorch_utils
from safetensors.torch import save_file


def _removed_in_transformers_5(*_args, **_kwargs):
    raise NotImplementedError("find_pruneable_heads_and_indices was removed in transformers 5")


# lxt 2.1's BERT patch imports a helper transformers 5 removed (see make_attnlrp_reference.py).
if not hasattr(transformers.pytorch_utils, "find_pruneable_heads_and_indices"):
    transformers.pytorch_utils.find_pruneable_heads_and_indices = _removed_in_transformers_5

import lxt  # noqa: E402
import transformers.models.esm.modeling_esm as modeling_esm  # noqa: E402
from lxt.efficient.patches import layer_norm_forward, wrap_attention_forward  # noqa: E402
from lxt.efficient.rules import identity_rule_implicit  # noqa: E402
from transformers import AutoTokenizer, EsmForMaskedLM  # noqa: E402

SEQUENCES = [
    "MKTAYIAKQRQISFVKSHFSRQ",
    # Human ubiquitin
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    # Protein G's B1 domain (GB1)
    "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
]
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def apply_attnlrp_rules():
    assert "eager" not in modeling_esm.ALL_ATTENTION_FUNCTIONS, "eager attention is registered; patch it there"
    modeling_esm.eager_attention_forward = wrap_attention_forward(modeling_esm.eager_attention_forward)
    gelu = modeling_esm.gelu
    modeling_esm.gelu = lambda x: identity_rule_implicit(gelu, x)
    torch.nn.LayerNorm.forward = layer_norm_forward


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_dir")
    p.add_argument("--plain-gradient", action="store_true", help="no AttnLRP rules: a negative control")
    a = p.parse_args()
    if not a.plain_gradient:
        apply_attnlrp_rules()
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    model = EsmForMaskedLM.from_pretrained(a.model_dir, attn_implementation="eager", dtype=torch.float32).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    hidden = model.config.hidden_size
    g = torch.Generator().manual_seed(7)
    head_weight = torch.randn(hidden, generator=g) / hidden ** 0.5
    head_bias = torch.tensor([0.25])
    aa_ids = torch.tensor([tok.convert_tokens_to_ids(c) for c in AMINO_ACIDS])

    captured = {}

    def capture(name):
        def hook(_module, _inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            if not out.requires_grad:  # the embeddings: frozen weights, integer input
                out = out.detach().requires_grad_(True)
                output = (out,) + tuple(output[1:]) if isinstance(output, tuple) else out
            else:
                out.retain_grad()
            captured[name] = out
            return output
        return hook

    model.esm.embeddings.register_forward_hook(capture(0))
    for i, layer in enumerate(model.esm.encoder.layer):
        layer.register_forward_hook(capture(i + 1))
    layers = len(model.esm.encoder.layer)

    def relevance(ids, output_fn):
        """Runs the model on ids, backpropagates output_fn(result) and returns per-token and per-layer relevance."""
        captured.clear()
        result = model(ids[None], output_hidden_states=True)
        value = output_fn(result)
        value.backward()
        per_layer = torch.stack([(captured[i] * captured[i].grad)[0].sum(-1) for i in range(layers + 1)])
        return value.detach(), per_layer[0].clone(), per_layer.contiguous()

    out = {"head_weight": head_weight.contiguous(), "head_bias": head_bias}
    for k, s in enumerate(SEQUENCES):
        ids = tok(s, return_tensors="pt")["input_ids"][0]
        pos = len(s) // 2 + 1  # a residue token in the middle
        masked = ids.clone()
        masked[pos] = tok.mask_token_id
        with torch.no_grad():
            logits = model(masked[None]).logits[0, pos]
        token = int(aa_ids[logits[aa_ids].argmax()])
        wild = int(ids[pos])
        if token == wild:  # explain a substitution: the likeliest other amino acid
            others = aa_ids[aa_ids != wild]
            token = int(others[logits[others].argmax()])
        head_pos = 2
        value, rel, per_layer = relevance(masked, lambda r: r.logits[0, pos, token])
        out[f"ids.{k}"] = ids.to(torch.int64)
        out[f"masked_ids.{k}"] = masked.to(torch.int64)
        out[f"target.{k}"] = torch.tensor([pos, token, wild, head_pos], dtype=torch.int64)
        out[f"mask_value.{k}"] = value.reshape(1)
        out[f"mask_relevance.{k}"] = rel
        out[f"mask_layers.{k}"] = per_layer
        _, out[f"mutation_relevance.{k}"], _ = relevance(masked, lambda r: r.logits[0, pos, token] - r.logits[0, pos, wild])
        _, out[f"protein_head_relevance.{k}"], _ = relevance(
            ids, lambda r: r.hidden_states[-1][0, 1:-1].mean(0) @ head_weight + head_bias[0])
        _, out[f"residue_head_relevance.{k}"], _ = relevance(ids, lambda r: r.hidden_states[-1][0, head_pos] @ head_weight + head_bias[0])
    name = "esm_plain_gradient.safetensors" if a.plain_gradient else "esm_attnlrp.safetensors"
    metadata = {"model": os.path.basename(os.path.normpath(a.model_dir)), "sequences": json.dumps(SEQUENCES),
                "lxt": getattr(lxt, "__version__", "2.1"), "torch": torch.__version__, "transformers": transformers.__version__,
                "rule": "gradient * input" if a.plain_gradient else "AttnLRP (lxt rules)"}
    save_file(out, os.path.join(a.model_dir, name), metadata=metadata)
    print("wrote", name, "for", len(SEQUENCES), "sequences of", metadata["model"])


if __name__ == "__main__":
    main()
