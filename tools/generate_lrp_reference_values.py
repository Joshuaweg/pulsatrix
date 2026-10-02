"""Offline, one-time reference-value generator for tests/lrp_reference_test.cpp (LRP-completeness
campaign Mission 3): pulsatrix's LRP checked against the reference LRP implementations.

Not run by CTest/CI -- this is a codegen tool, not a test dependency. Run it manually inside the
pinned image built from tools/lrp_reference.Dockerfile:

    docker build -t pulsatrix-lrpref:latest -f tools/lrp_reference.Dockerfile tools
    docker run --rm -v "$PWD":/w -w /w pulsatrix-lrpref:latest \
        python3 tools/generate_lrp_reference_values.py

and transcribe the printed C++ initializer lists into tests/lrp_reference_test.cpp as hardcoded
targets. Zero live Python dependency at C++ test/build time.

Versions the committed values were generated with (pinned in tools/lrp_reference.Dockerfile):
    Python 3.12, torch 2.14.1+cpu, torchvision 0.29.1+cpu, numpy 2.5.2, zennit 1.0.0, lxt 2.1

Part A -- Zennit 1.0.0 (zennit.attribution.Gradient + a composite, attr_output = one-hot of one
target per row; the attribution is then exactly the LRP relevance at the input). Two models:
    MLP: Linear(6,8) -> ReLU -> Linear(8,5) -> ReLU -> Linear(5,3), input (2, 6)
    CNN: Conv2d(1,2,3) -> ReLU -> Conv2d(2,3,2) -> ReLU -> Flatten -> Linear(27,4), input (2,1,6,6)
Rule sets: uniform Epsilon (1e-6 and 0.25), uniform ZPlus, uniform AlphaBeta(2,1), uniform
Gamma(0.25) -- each via a LayerMapComposite with Pass on ReLU, exactly as Zennit's own presets do
(layer_map_base) -- and the presets EpsilonPlus, EpsilonAlpha2Beta1, EpsilonGammaBox(-3, 3).
Every stabilizer is Zennit's default 1e-6 (pulsatrix: LRPRuleConfig::epsilon).

Part B -- LXT 2.1 explicit AttnLRP (lxt.explicit.functional / lxt.explicit.rules). LXT's own
model support is HF-transformers monkey patching (lxt.efficient), which cannot load a pulsatrix
module, so the same computation is written out op-by-op with LXT's explicit rule functions --
every relevance-carrying op is an LXT rule, nothing hand-derived:
    linear_epsilon (Q/K/V/O, SwiGLU projections, classifier), matmul (Q@K^T and Attn@V:
    epsilon + uniform split), softmax with temperature sqrt(head_dim) (AttnLRP Prop. 3.1 on the
    scaled scores), rms_norm_identity, add2 (residuals), mul2 (SwiGLU gate * up),
    rules.identity on SiLU. Reshapes / head permutes are plain autograd (exact for relevance).
Two models, both followed by Flatten -> Linear(L*d_model, 3) to give pulsatrix a rank-2 output:
    MHA:   MultiHeadAttentionModule(d_model=4, heads=2, no RoPE, no QK-Norm), input (2, 3, 4)
    Block: TransformerBlock(d_model=4, heads=2, d_ff=6, no RoPE, no QK-Norm), input (2, 3, 4)
All LXT epsilons are set to 1e-6 (pulsatrix: LRPRuleConfig{1e-6} with
epsilon_bias_in_denominator = true, because LXT's linear_epsilon keeps the bias in z).

Deterministic, exactly shared parameters: every weight/bias/input is det(n, mul, add, div), i.e.
float32(((i * mul + add) % 17) - 8) / float32(div) for i in [0, n) -- the same integer formula
and a single IEEE float32 division on both sides, so the C++ test (Det() there) rebuilds
bit-identical tensors without a page of literals.

IMPORTANT layout gotcha (same as tools/generate_captum_reference_values.py): pulsatrix's
LinearModule weight is (in_features, out_features) row-major -- the transpose of
torch.nn.Linear.weight. Every linear weight below is generated in pulsatrix's (in, out) order and
.t()'d into torch. Conv2DModule's kernel is (out, in, kh, kw) like torch's, its bias (out,), and
it is stride 1 / no padding like nn.Conv2d's defaults -- no conversion there. FlattenModule
flattens (C, H, W) row-major like nn.Flatten.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from lxt.explicit import functional as lf
from lxt.explicit import rules as lrules
from zennit.attribution import Gradient
from zennit.composites import (EpsilonAlpha2Beta1, EpsilonGammaBox, EpsilonPlus,
                               LayerMapComposite)
from zennit.rules import AlphaBeta, Epsilon, Gamma, Pass, ZPlus
from zennit.types import Activation, Convolution

torch.set_grad_enabled(True)


def det(n, mul, add, div):
    """float32(((i*mul + add) % 17) - 8) / float32(div), computed as one float32 division."""
    ints = torch.tensor([float((i * mul + add) % 17 - 8) for i in range(n)], dtype=torch.float32)
    return ints / torch.tensor(float(div), dtype=torch.float32)


def linear(in_f, out_f, mul, add, div, bias_add, bias_div):
    """nn.Linear whose weight is det() in pulsatrix's (in, out) order, transposed into torch."""
    m = nn.Linear(in_f, out_f)
    with torch.no_grad():
        m.weight.copy_(det(in_f * out_f, mul, add, div).reshape(in_f, out_f).t())
        m.bias.copy_(det(out_f, 3, bias_add, bias_div))
    return m


def conv(in_c, out_c, k, mul, add, div, bias_add, bias_div):
    m = nn.Conv2d(in_c, out_c, k)
    with torch.no_grad():
        m.weight.copy_(det(out_c * in_c * k * k, mul, add, div).reshape(out_c, in_c, k, k))
        m.bias.copy_(det(out_c, 3, bias_add, bias_div))
    return m


def mlp():
    return nn.Sequential(
        linear(6, 8, 5, 1, 7, 2, 11),
        nn.ReLU(),
        linear(8, 5, 3, 1, 9, 6, 13),
        nn.ReLU(),
        linear(5, 3, 13, 1, 6, 10, 7),
    )


def cnn():
    return nn.Sequential(
        conv(1, 2, 3, 5, 2, 6, 4, 9),
        nn.ReLU(),
        conv(2, 3, 2, 7, 5, 8, 7, 10),
        nn.ReLU(),
        nn.Flatten(),
        linear(27, 4, 11, 3, 13, 1, 7),
    )


MLP_INPUT = det(12, 5, 3, 5).reshape(2, 6)
MLP_TARGETS = [2, 1]
CNN_INPUT = det(72, 7, 1, 3).reshape(2, 1, 6, 6)
CNN_TARGETS = [1, 3]


def uniform(rule):
    """Zennit's own preset structure (Pass on activations), with one rule on every affine layer."""
    return LayerMapComposite(layer_map=[(Activation, Pass()), ((Convolution, nn.Linear), rule)])


def zennit_cases(x):
    low = torch.full_like(x, -3.0)
    high = torch.full_like(x, 3.0)
    return [
        ("Epsilon1e6", uniform(Epsilon(epsilon=1e-6))),
        ("Epsilon025", uniform(Epsilon(epsilon=0.25))),
        ("ZPlus", uniform(ZPlus(stabilizer=1e-6))),
        ("AlphaBeta21", uniform(AlphaBeta(alpha=2.0, beta=1.0, stabilizer=1e-6))),
        ("Gamma025", uniform(Gamma(gamma=0.25, stabilizer=1e-6))),
        ("EpsilonPlus", EpsilonPlus()),
        ("EpsilonAlpha2Beta1", EpsilonAlpha2Beta1()),
        ("EpsilonGammaBox", EpsilonGammaBox(low=low, high=high)),
    ]


def one_hot(targets, classes):
    return torch.eye(classes)[targets]


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
    print(line[:-1])
    print("};")


def zennit_part(prefix, model, x, targets):
    model.eval()
    classes = model(x).shape[1]
    for name, composite in zennit_cases(x):
        with Gradient(model=model, composite=composite) as attributor:
            _, relevance = attributor(x.clone(), one_hot(targets, classes))
        emit(prefix + name, relevance)


# ---- Part B: LXT explicit AttnLRP --------------------------------------------------------------
EPS = 1e-6
D_MODEL, HEADS, D_FF, SEQ = 4, 2, 6, 3
HEAD_DIM = D_MODEL // HEADS
ATTN_INPUT = det(2 * SEQ * D_MODEL, 5, 7, 4).reshape(2, SEQ, D_MODEL)
ATTN_TARGETS = [0, 2]


def lin_params(in_f, out_f, mul, add, div, bias_add, bias_div):
    """(weight in torch's (out, in) layout, bias) for lxt linear_epsilon -- see layout gotcha."""
    w = det(in_f * out_f, mul, add, div).reshape(in_f, out_f).t().contiguous()
    return w, det(out_f, 3, bias_add, bias_div)


MHA_PARAMS = {
    "q": lin_params(4, 4, 5, 1, 6, 2, 11),
    "k": lin_params(4, 4, 7, 3, 5, 5, 13),
    "v": lin_params(4, 4, 11, 6, 7, 8, 9),
    "o": lin_params(4, 4, 13, 2, 8, 1, 10),
}
BLOCK_NORM1_GAMMA = det(4, 3, 14, 8) + 1.0
BLOCK_NORM2_GAMMA = det(4, 5, 11, 8) + 1.0
SWIGLU_PARAMS = {
    "gate": lin_params(4, 6, 5, 9, 6, 4, 12),
    "up": lin_params(4, 6, 7, 2, 7, 7, 10),
    "down": lin_params(6, 4, 3, 8, 6, 9, 11),
}
HEAD_PARAMS = lin_params(SEQ * D_MODEL, 3, 11, 5, 9, 3, 7)


def mha_lxt(h):
    n = h.shape[0]
    q = lf.linear_epsilon(h, *MHA_PARAMS["q"], epsilon=EPS)
    k = lf.linear_epsilon(h, *MHA_PARAMS["k"], epsilon=EPS)
    v = lf.linear_epsilon(h, *MHA_PARAMS["v"], epsilon=EPS)

    def split(t):  # (N, L, d) -> (N, H, L, D)
        return t.reshape(n, SEQ, HEADS, HEAD_DIM).permute(0, 2, 1, 3)

    q, k, v = split(q), split(k), split(v)
    scores = lf.matmul(q, k.transpose(-1, -2), epsilon=EPS)       # raw Q @ K^T
    attn = lf.softmax(scores, -1, temperature=math.sqrt(HEAD_DIM))  # softmax(raw / sqrt(D))
    context = lf.matmul(attn, v, epsilon=EPS)
    merged = context.permute(0, 2, 1, 3).reshape(n, SEQ, D_MODEL)
    return lf.linear_epsilon(merged, *MHA_PARAMS["o"], epsilon=EPS)


def swiglu_lxt(h):
    gate = lf.linear_epsilon(h, *SWIGLU_PARAMS["gate"], epsilon=EPS)
    gate = lrules.identity(F.silu, gate)
    up = lf.linear_epsilon(h, *SWIGLU_PARAMS["up"], epsilon=EPS)
    return lf.linear_epsilon(lf.mul2(gate, up), *SWIGLU_PARAMS["down"], epsilon=EPS)


def block_lxt(x):
    y1 = lf.add2(x, mha_lxt(lf.rms_norm_identity(x, BLOCK_NORM1_GAMMA, 1e-6)), epsilon=EPS)
    return lf.add2(y1, swiglu_lxt(lf.rms_norm_identity(y1, BLOCK_NORM2_GAMMA, 1e-6)), epsilon=EPS)


def lxt_part(prefix, body):
    x = ATTN_INPUT.clone().requires_grad_(True)
    hidden = body(x)
    logits = lf.linear_epsilon(hidden.reshape(x.shape[0], -1), *HEAD_PARAMS, epsilon=EPS)
    logits.backward(one_hot(ATTN_TARGETS, 3))
    emit(prefix + "Logits", logits)
    emit(prefix + "Relevance", x.grad)


def main():
    print("// ---- Part A: Zennit 1.0.0 ----")
    zennit_part("Mlp", mlp(), MLP_INPUT, MLP_TARGETS)
    zennit_part("Cnn", cnn(), CNN_INPUT, CNN_TARGETS)
    print("// ---- Part B: LXT 2.1 explicit AttnLRP ----")
    lxt_part("Mha", mha_lxt)
    lxt_part("Block", block_lxt)


if __name__ == "__main__":
    main()
