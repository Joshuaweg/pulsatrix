"""Offline, one-time generator for tests/fixtures/encoder/ (PLM-1): one transformer encoder layer
from transformers itself, in each layout EncoderBlock supports, with its input, output and the
gradient of a fixed projection of the output with respect to the input.

Not run by CTest/CI. Run it in the image of tools/lrp_reference.Dockerfile (pulsatrix-lrpref) from
the repository root:

    python3 tools/generate_encoder_reference.py

Versions the committed values were generated with: Python 3.12, torch 2.14.1, transformers 5.18.0.

Layers, each wide 32 with 4 heads and an MLP of 48, on a (1, 6, 32) input:
    esm_layer:  EsmLayer (ESM-2): pre-LayerNorm (eps 1e-5), rotary position embeddings in the
                rotate-half layout, Q/K/V and output biases, exact-GELU MLP.
    bert_layer: BertLayer: post-LayerNorm (eps 1e-12), no position embeddings inside the layer,
                exact-GELU MLP.

Each directory holds model.safetensors with the layer's own Hugging Face parameter names plus
"ref.input", "ref.output", "ref.output_grad" (the projection) and "ref.input_grad", all float32.
Weights are drawn with seed 0 (std 0.2); LayerNorm weights around 1.
"""

import os

import torch
from safetensors.torch import save_file
from transformers import BertConfig, EsmConfig
from transformers.models.bert.modeling_bert import BertLayer
from transformers.models.esm.modeling_esm import EsmLayer, EsmRotaryEmbedding

OUT = "tests/fixtures/encoder"
COMMON = dict(hidden_size=32, num_attention_heads=4, intermediate_size=48, hidden_act="gelu",
              hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)


def randomize(module, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in module.named_parameters():
            noise = torch.randn(p.shape, generator=g) * 0.2
            p.copy_(1.0 + noise if "LayerNorm.weight" in name else noise)


def dump(name, layer, rotary=None):
    layer.eval()
    randomize(layer, 0)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 6, 32, generator=g).requires_grad_(True)
    kwargs = {}
    if rotary is not None:  # ESM computes cos/sin once per model and hands them to every layer
        kwargs["position_embeddings"] = rotary(x, torch.arange(x.shape[1]).unsqueeze(0))
    out = layer(x, **kwargs)
    y = out[0] if isinstance(out, tuple) else out
    r = torch.randn(y.shape, generator=g)
    (y * r).sum().backward()
    tensors = {k: v.detach().contiguous() for k, v in layer.state_dict().items() if v.dtype == torch.float32}
    tensors.update({"ref.input": x.detach(), "ref.output": y.detach(), "ref.output_grad": r, "ref.input_grad": x.grad.detach()})
    os.makedirs(f"{OUT}/{name}", exist_ok=True)
    save_file(tensors, f"{OUT}/{name}/model.safetensors")
    print(name, sorted(k for k in tensors if not k.startswith("ref.")))


torch.manual_seed(0)
esm = EsmConfig(vocab_size=33, position_embedding_type="rotary", layer_norm_eps=1e-5, **COMMON)
dump("esm_layer", EsmLayer(esm), EsmRotaryEmbedding(config=esm))
dump("bert_layer", BertLayer(BertConfig(layer_norm_eps=1e-12, **COMMON)))
