"""Offline, one-time reference-value generator for tests/attention_reference_test.cpp (LLM-1):
pulsatrix's MultiHeadAttentionModule checked against Hugging Face Transformers' own attention
classes, on the configurations real checkpoints use.

Not run by CTest/CI -- this is a codegen tool, not a test dependency. Run it in the same pinned
image as tools/generate_lrp_reference_values.py (tools/lrp_reference.Dockerfile already installs
transformers):

    docker build -t pulsatrix-lrpref:latest -f tools/lrp_reference.Dockerfile tools
    docker run --rm -v "$PWD":/w -w /w pulsatrix-lrpref:latest \
        python3 tools/generate_attention_reference_values.py

and transcribe the printed C++ initializer lists into tests/attention_reference_test.cpp.

Versions the committed values were generated with: Python 3.12, torch 2.14.1, transformers 5.18.0.

Three cases, each an (N=2, L=5) batch through one attention layer with eager attention:
    Llama: LlamaAttention, d_model 8, 4 query heads, 2 K/V heads, head_dim 4 (so the query
           width 16 differs from d_model), no biases, rope_theta 100, causal, the second
           sequence right-padded by 2.
    Qwen2: Qwen2Attention, d_model 6, 3 query heads, 1 K/V head (multi-query), head_dim 4, Q/K/V
           biases and no output bias, rope_theta 1000, causal, the first sequence left-padded
           by 1 (so its query 0 has every key masked), positions starting at 3.
    Qwen3: Qwen3Attention, d_model 8, 2 query heads, 1 K/V head, head_dim 6, QK-Norm with
           non-unit gammas and eps 1e-6, no biases, rope_theta 10000, causal, no padding.

For each case it prints the output, and the gradient of sum(output * G) with respect to the
input, where G is det() times the query keep mask: padding queries carry no loss, as in
training, because Hugging Face's additive mask lets gradient leak through a fully masked row.

Deterministic, exactly shared parameters: every weight/bias/input is det(n, mul, add, div), the
same formula as tools/generate_lrp_reference_values.py, so the C++ test rebuilds bit-identical
tensors. pulsatrix's LinearModule weight is (in, out); torch's nn.Linear.weight is (out, in), so
every weight is generated in pulsatrix's order and .t()'d into torch.
"""

import torch
from transformers import LlamaConfig, Qwen2Config, Qwen3Config
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2RotaryEmbedding
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3RotaryEmbedding

torch.set_grad_enabled(True)


def det(n, mul, add, div):
    """float32(((i*mul + add) % 17) - 8) / float32(div), computed as one float32 division."""
    ints = torch.tensor([float((i * mul + add) % 17 - 8) for i in range(n)], dtype=torch.float32)
    return ints / torch.tensor(float(div), dtype=torch.float32)


def set_linear(lin, in_f, out_f, mul, add, div, bias=None):
    with torch.no_grad():
        lin.weight.copy_(det(in_f * out_f, mul, add, div).reshape(in_f, out_f).t())
        if bias is not None:
            lin.bias.copy_(det(out_f, 3, bias[0], bias[1]))


def additive_mask(keep, causal=True):
    """(N, L) keep -> (N, 1, L, L) additive mask, Hugging Face's eager-attention form."""
    n, l = keep.shape
    allowed = keep[:, None, None, :].bool().expand(n, 1, l, l)
    if causal:
        allowed = allowed & torch.ones(l, l, dtype=torch.bool).tril()[None, None]
    return torch.zeros(n, 1, l, l).masked_fill(~allowed, torch.finfo(torch.float32).min)


def run(name, attn, rotary, d_model, keep, offset, x_det, g_det):
    n, l = keep.shape
    x = det(n * l * d_model, *x_det).reshape(n, l, d_model).requires_grad_(True)
    position_ids = torch.arange(offset, offset + l)[None].expand(n, l)
    cos, sin = rotary(x, position_ids)
    out, _ = attn(x, position_embeddings=(cos, sin), attention_mask=additive_mask(keep))
    g = det(n * l * d_model, *g_det).reshape(n, l, d_model) * keep[:, :, None]
    (out * g).sum().backward()
    emit(name + "Output", out)
    emit(name + "InputGrad", x.grad)


def cpp_float(v):
    """%.9g round-trips float32; '0' / '-0' / '2' need a decimal point to be C++ float literals."""
    text = "%.9g" % (v + 0.0)  # + 0.0 turns -0.0 into 0.0
    return (text if any(c in text for c in ".e") else text + ".0") + "f"


def emit(name, values):
    """Prints a C++ initializer list, wrapped to the repo's 120-column limit."""
    print("const std::vector<float> k%s = {" % name)
    line = "   "
    for v in values.detach().reshape(-1).tolist():
        item = " " + cpp_float(v) + ","
        if len(line) + len(item) > 116:
            print(line)
            line = "   "
        line += item
    print(line)
    print("};")


def llama():
    cfg = LlamaConfig(hidden_size=8, num_attention_heads=4, num_key_value_heads=2, head_dim=4,
                      attention_bias=False, rope_theta=100.0, max_position_embeddings=64)
    cfg._attn_implementation = "eager"
    attn = LlamaAttention(cfg, layer_idx=0).eval()
    set_linear(attn.q_proj, 8, 16, 5, 1, 7)
    set_linear(attn.k_proj, 8, 8, 3, 2, 9)
    set_linear(attn.v_proj, 8, 8, 7, 3, 8)
    set_linear(attn.o_proj, 16, 8, 11, 4, 10)
    keep = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.float32)
    run("Llama", attn, LlamaRotaryEmbedding(cfg), 8, keep, 0, (5, 3, 4), (7, 2, 5))


def qwen2():
    cfg = Qwen2Config(hidden_size=6, num_attention_heads=3, num_key_value_heads=1, head_dim=4,
                      rope_theta=1000.0, max_position_embeddings=64, use_sliding_window=False)
    cfg._attn_implementation = "eager"
    attn = Qwen2Attention(cfg, layer_idx=0).eval()
    set_linear(attn.q_proj, 6, 12, 5, 2, 6, bias=(1, 5))
    set_linear(attn.k_proj, 6, 4, 3, 1, 7, bias=(4, 6))
    set_linear(attn.v_proj, 6, 4, 7, 5, 9, bias=(2, 7))
    set_linear(attn.o_proj, 12, 6, 13, 3, 11)
    keep = torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.float32)
    run("Qwen2", attn, Qwen2RotaryEmbedding(cfg), 6, keep, 3, (3, 1, 4), (5, 4, 6))


def qwen3():
    cfg = Qwen3Config(hidden_size=8, num_attention_heads=2, num_key_value_heads=1, head_dim=6,
                      attention_bias=False, rope_theta=10000.0, rms_norm_eps=1e-6,
                      max_position_embeddings=64, use_sliding_window=False)
    cfg._attn_implementation = "eager"
    attn = Qwen3Attention(cfg, layer_idx=0).eval()
    set_linear(attn.q_proj, 8, 12, 5, 1, 6)
    set_linear(attn.k_proj, 8, 6, 3, 4, 7)
    set_linear(attn.v_proj, 8, 6, 7, 2, 8)
    set_linear(attn.o_proj, 12, 8, 11, 5, 9)
    with torch.no_grad():
        attn.q_norm.weight.copy_(1.0 + det(6, 5, 1, 20))
        attn.k_norm.weight.copy_(1.0 + det(6, 7, 3, 20))
    keep = torch.ones(2, 5)
    run("Qwen3", attn, Qwen3RotaryEmbedding(cfg), 8, keep, 0, (7, 1, 4), (3, 5, 5))


if __name__ == "__main__":
    llama()
    qwen2()
    qwen3()
