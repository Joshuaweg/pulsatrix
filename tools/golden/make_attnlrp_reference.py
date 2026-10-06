"""Writes AttnLRP reference relevance for LLM-7: LXT's own AttnLRP (lxt.efficient, the
transformers monkey patch) on a Hugging Face model, to compare pulsatrix's relevance against
(tests/attnlrp_parity_test.cpp, pulsatrix_attnlrp).

    docker run --rm -v "$PWD":/w -v ~/.cache/pulsatrix/golden:/golden -w /w pulsatrix-lrpref:latest \
        python3 tools/golden/make_attnlrp_reference.py /golden/SmolLM2-135M

reads the token ids from the model directory's golden.safetensors (tools/golden/make_golden.py),
so both harnesses explain the same tokens, and writes attnlrp.safetensors beside it:
    ids.<k>             I64 [L]            the tokens, copied from golden.safetensors
    target.<k>          I64 [1]            the explained token: the argmax at the last position
    relevance.<k>       F32 [L]            relevance of each input token (embedding * gradient,
                                           summed over the hidden size)
    key_relevance.<k>   F32 [layers, Hkv, L]  relevance at each layer's k_proj output, per KV head
    value_relevance.<k> F32 [layers, Hkv, L]  the same at v_proj
Relevance at the output is the explained logit itself (pulsatrix's LRPSeed::Logit): in gradient *
input that is a unit gradient, logit.backward(). LXT's README calls logit.backward(logit), which
scales every relevance by the logit once more.

--plain-gradient skips LXT's patches and writes plain gradient * input instead: a negative
control that pulsatrix's AttnLRP must not match.

LXT's AttnLRP is gradient * input with modified gradients: the identity rule on RMSNorm (and
QK-Norm) and on SiLU, the uniform rule at the gate * up product and at both attention matmuls
(Q and K / 4, V / 2), and plain gradients elsewhere, which is the epsilon rule with epsilon -> 0.
Needs the pulsatrix-lrpref image (tools/lrp_reference.Dockerfile: torch, transformers, lxt 2.1).
"""

import argparse
import importlib
import os

import torch
import transformers
import transformers.pytorch_utils
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM


def _removed_in_transformers_5(*_args, **_kwargs):
    raise NotImplementedError("find_pruneable_heads_and_indices was removed in transformers 5")


# lxt 2.1 imports every model's patch on load, and its BERT patch imports a helper transformers 5
# removed. Only the Llama, Qwen2 and Qwen3 patches run here, so the missing name gets a stub.
if not hasattr(transformers.pytorch_utils, "find_pruneable_heads_and_indices"):
    transformers.pytorch_utils.find_pruneable_heads_and_indices = _removed_in_transformers_5

import lxt  # noqa: E402
from lxt.efficient import monkey_patch  # noqa: E402
from lxt.efficient.models import get_default_map  # noqa: E402
from lxt.efficient.patches import wrap_attention_forward  # noqa: E402


def patch_eager_attention(modeling):
    """LXT's attention patch (Q and K / 4, V / 2), on the eager function only.

    lxt 2.1's patch_attention also replaces ALL_ATTENTION_FUNCTIONS with a plain dict, which
    transformers 5 can't dispatch through (it calls .get_interface). With
    attn_implementation="eager", transformers 5 falls back to the module's own
    eager_attention_forward, so wrapping that one function with LXT's wrapper is the whole rule.
    """
    assert "eager" not in modeling.ALL_ATTENTION_FUNCTIONS, "eager attention is registered; patch it there"
    modeling.eager_attention_forward = wrap_attention_forward(modeling.eager_attention_forward)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_dir")
    p.add_argument("--golden", default=None, help="token ids (default: MODEL_DIR/golden.safetensors)")
    p.add_argument("--out", default=None, help="default: MODEL_DIR/attnlrp.safetensors")
    p.add_argument("--plain-gradient", action="store_true", help="no LXT patches: a negative control")
    a = p.parse_args()
    golden = a.golden or os.path.join(a.model_dir, "golden.safetensors")
    out = a.out or os.path.join(a.model_dir, "attnlrp.safetensors")

    config = AutoConfig.from_pretrained(a.model_dir)
    modeling = importlib.import_module("transformers.models.%s.modeling_%s" % (config.model_type, config.model_type))
    if not a.plain_gradient:
        patch_map = dict(get_default_map(modeling))
        patch_map[modeling] = patch_eager_attention
        monkey_patch(modeling, patch_map)
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=torch.float32, attn_implementation="eager").eval()
    for param in model.parameters():
        param.requires_grad_(False)

    layers = model.model.layers
    kv_heads = config.num_key_value_heads
    captured = {}

    def capture(name):
        def hook(_module, _inputs, output):
            output.retain_grad()
            captured[name] = output
        return hook

    for i, layer in enumerate(layers):
        layer.self_attn.k_proj.register_forward_hook(capture("k.%d" % i))
        layer.self_attn.v_proj.register_forward_hook(capture("v.%d" % i))

    def per_head(name, length):
        t = captured[name]
        r = (t * t.grad)[0].reshape(length, kv_heads, -1).sum(-1)  # (L, Hkv)
        return r.transpose(0, 1)  # (Hkv, L)

    tensors = {}
    with safe_open(golden, "pt") as f:
        k = 0
        while "ids.%d" % k in f.keys():
            ids = f.get_tensor("ids.%d" % k)
            length = ids.shape[0]
            embeds = model.get_input_embeddings()(ids[None]).detach().requires_grad_(True)
            logits = model(inputs_embeds=embeds).logits[0, -1]
            target = int(logits.argmax())
            logits[target].backward()
            tensors["ids.%d" % k] = ids
            tensors["target.%d" % k] = torch.tensor([target], dtype=torch.int64)
            tensors["relevance.%d" % k] = (embeds * embeds.grad)[0].sum(-1).contiguous()
            tensors["key_relevance.%d" % k] = torch.stack([per_head("k.%d" % i, length) for i in range(len(layers))]).contiguous()
            tensors["value_relevance.%d" % k] = torch.stack([per_head("v.%d" % i, length) for i in range(len(layers))]).contiguous()
            k += 1
        metadata = dict(f.metadata() or {})
    metadata.update({"lxt": getattr(lxt, "__version__", "2.1"), "torch": torch.__version__,
                     "transformers": transformers.__version__, "rule": "gradient * input" if a.plain_gradient else "AttnLRP (lxt.efficient)",
                     "seed": "logit", "texts": str(k)})
    save_file(tensors, out, metadata=metadata)
    print("wrote AttnLRP relevance for %d sequences of %s" % (k, metadata.get("model", a.model_dir)))


if __name__ == "__main__":
    main()
