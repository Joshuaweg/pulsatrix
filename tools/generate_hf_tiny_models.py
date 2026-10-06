"""Offline, one-time generator for tests/fixtures/hf_tiny/ and tests/causal_lm_test.cpp's reference
values (IO-4): tiny randomly initialized Hugging Face models saved with save_pretrained, and their
logits and greedy generations from transformers itself.

Not run by CTest/CI. Run it in the image of tools/lrp_reference.Dockerfile (or any environment with
torch and transformers) from the repository root:

    python3 tools/generate_hf_tiny_models.py

Versions the committed values were generated with: Python 3.12, torch 2.14.1, transformers 5.18.0.

Three models, each 2 layers wide 32 with a 64-token vocabulary, covering the layouts pulsatrix maps:
    llama: grouped-query (4 heads / 2 K/V), tied embeddings, rope_theta 1000, rms_norm_eps 1e-5,
           saved in shards (max_shard_size) so the sharded index is exercised.
    qwen2: multi-query (4 / 1), Q/K/V biases, an untied lm_head (transposed on load).
    qwen3: head_dim 12 (not 32 / 4), QK-Norm, tied embeddings.
Weights are drawn with seed 0 (std 0.2), rounded to bf16 and saved as bf16, so the file holds
exactly the values the float32 reference forward pass uses.
"""

import os

import torch
from transformers import (LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM, Qwen3Config,
                          Qwen3ForCausalLM)

OUT = "tests/fixtures/hf_tiny"
TOKENS = [[3, 17, 42, 5, 60, 8]]
COMMON = dict(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
              max_position_embeddings=128, bos_token_id=1, eos_token_id=2)


def build(cls, config):
    torch.manual_seed(0)
    model = cls(config).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.copy_((torch.randn_like(p) * 0.2).to(torch.bfloat16).float())
        for name, p in model.named_parameters():
            if "norm" in name:  # norms near 1, as trained models have them
                p.copy_((1.0 + torch.randn_like(p) * 0.1).to(torch.bfloat16).float())
    return model


def cpp_float(v):
    text = "%.9g" % (float(v) + 0.0)
    return (text if any(c in text for c in ".e") else text + ".0") + "f"


def emit(name, values):
    print("const std::vector<float> k%s = {" % name)
    line = "   "
    for v in torch.as_tensor(values).reshape(-1).tolist():
        item = " " + cpp_float(v) + ","
        if len(line) + len(item) > 116:
            print(line)
            line = "   "
        line += item
    print(line)
    print("};")


def run(tag, cls, config, shard=None):
    config._attn_implementation = "eager"
    model = build(cls, config)
    with torch.no_grad():
        logits = model(torch.tensor(TOKENS)).logits
        generated = model.generate(torch.tensor(TOKENS), max_new_tokens=8, do_sample=False, eos_token_id=None,
                                   pad_token_id=0)
    path = os.path.join(OUT, tag)
    kwargs = {"max_shard_size": shard} if shard else {}
    model.to(torch.bfloat16).save_pretrained(path, safe_serialization=True, **kwargs)
    emit(tag.capitalize() + "Logits", logits)
    print("const std::vector<int64_t> k%sGenerated = {%s};" % (tag.capitalize(),
                                                             ", ".join(str(t) for t in generated[0, 6:].tolist())))


if __name__ == "__main__":
    run("llama", LlamaForCausalLM, LlamaConfig(**COMMON, num_key_value_heads=2, rope_theta=1000.0, rms_norm_eps=1e-5,
                                               tie_word_embeddings=True), shard="20KB")
    run("qwen2", Qwen2ForCausalLM, Qwen2Config(**COMMON, num_key_value_heads=1, rope_theta=10000.0, rms_norm_eps=1e-6,
                                               tie_word_embeddings=False, use_sliding_window=False))
    run("qwen3", Qwen3ForCausalLM, Qwen3Config(**COMMON, num_key_value_heads=2, head_dim=12, rope_theta=10000.0,
                                               rms_norm_eps=1e-6, tie_word_embeddings=True, use_sliding_window=False))
