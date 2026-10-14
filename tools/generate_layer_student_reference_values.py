"""Offline, one-time reference-value generator for tests/layer_student_test.cpp and
tests/born_again_test.cpp (KD-4): pulsatrix's HiddenStateCosineLoss, EnsembleLogProbabilities and
PermuteDarkKnowledge checked against numpy, in float64.

Not run by CTest/CI. Run it with numpy installed and transcribe the printed values into the tests:

    python3 tools/generate_layer_student_reference_values.py

Versions the committed values were generated with: Python 3.14.4, numpy 2.5.4.

Inputs are float32(sin(a k + b)) of the flat index k, with (a, b) per tensor as below.

  - The cosine loss is DistilBERT's (huggingface/transformers, examples/research_projects/
    distillation/distiller.py): CosineEmbeddingLoss(reduction="mean") with target 1 over the
    hidden states the attention mask keeps, that is mean over kept rows of 1 - cos(s, t). The
    gradient is a central difference of the loss.
  - The ensemble is log(mean_g softmax(l_g)) row by row.
  - The permutation is Fisher-Yates over each row's non-maximal entries, with j = next() % i from
    PortableRng(seed): ((z ^ z >> 30) * C1, ...) SplitMix64, as src/portable_random.hpp.
"""

import numpy as np

MASK = (1 << 64) - 1


class PortableRng:
    def __init__(self, seed):
        self.state = seed

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)


def features(shape, a, b):
    k = np.arange(int(np.prod(shape)), dtype=np.float64)
    return np.sin(a * k + b).astype(np.float32).astype(np.float64).reshape(shape)


def central(f, x, h=1e-6):
    g = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        up, down = x.copy(), x.copy()
        up[i] += h
        down[i] -= h
        g[i] = (f(up) - f(down)) / (2 * h)
    return g


def vector(name, values):
    print(f"const std::vector<double> k{name} = {{" + ", ".join(f"{v:.12f}" for v in np.ravel(values)) + "};")


# Cosine loss: (2, 3, 5) hidden states, row 4 (the second sequence's second token) masked out.
s = features((2, 3, 5), 0.7, 0.3)
t = features((2, 3, 5), 0.45, 0.1)
mask = np.array([1, 1, 1, 1, 0, 1], dtype=np.float64)


def cosine_loss(x):
    a, b = x.reshape(-1, 5), t.reshape(-1, 5)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return ((1 - cos) * mask).sum() / mask.sum()


print(f"const double kCosineLoss = {cosine_loss(s):.12f};")
vector("CosineGrad", central(cosine_loss, s))

# Ensemble of three generations' (2, 2, 4) logits.
gens = [features((2, 2, 4), 0.8, 0.1) * 3, features((2, 2, 4), 0.3, 0.9) * 3, features((2, 2, 4), 1.3, 0.4) * 3]


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


vector("Ensemble", np.log(np.mean([softmax(g) for g in gens], axis=0)))

# The permutation of (3, 5) logits with seed 11.
logits = features((3, 5), 0.9, 0.2) * 4
out = logits.copy().ravel()
rng = PortableRng(11)
for r in range(3):
    o = r * 5
    top = o + int(np.argmax(out[o:o + 5]))
    others = [o + k for k in range(5) if o + k != top]
    for i in range(len(others), 1, -1):
        j = rng.next() % i
        out[others[i - 1]], out[others[j]] = out[others[j]], out[others[i - 1]]
vector("Permuted", out)
