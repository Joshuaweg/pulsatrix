"""Offline, one-time reference-value generator for tests/feature_distillation_test.cpp (KD-3):
pulsatrix's HintLoss and AttentionTransferLoss checked against numpy, in float64.

Not run by CTest/CI. Run it with numpy installed and transcribe the printed values into
tests/feature_distillation_test.cpp:

    python3 tools/generate_feature_distillation_reference_values.py

Versions the committed values were generated with: Python 3.14.4, numpy 2.5.4.

Features are float32(sin(a k + b)) of the flat index k, with (a, b) per tensor as below.

  - Hint loss (Romero et al. 2015): mean((x W + b - t)^2), W (in, out) and b drawn as pulsatrix
    draws them: PortableRng(seed 3), uniform = ((bits >> 11) + 0.5) / 2^53, each value
    (2 u - 1) / sqrt(in), the weights in row-major order and then the bias, rounded to float32.
  - Attention transfer, the authors' code (github.com/szagoruyko/attention-transfer, utils.py):
        at(x) = F.normalize(x.pow(2).mean(1).view(x.size(0), -1))
        at_loss(x, y) = (at(x) - at(y)).pow(2).mean()
    and, for p = 3, |x|^3 in place of x.pow(2).

The gradients are central differences of the loss.
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

    def uniform(self):
        return ((self.next() >> 11) + 0.5) / float(1 << 53)


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


def show(name, value, grad):
    print(f"const double k{name}Loss = {value:.12f};")
    print(f"const std::vector<double> k{name}Grad = {{" + ", ".join(f"{v:.12f}" for v in grad.ravel()) + "};")


# Hint loss: student (2, 3, 4), teacher (2, 3, 5).
student = features((2, 3, 4), 0.7, 0.3)
teacher = features((2, 3, 5), 0.45, 0.1)
rng = PortableRng(3)
bound = 1 / np.sqrt(4)
w = np.array([(2 * rng.uniform() - 1) * bound for _ in range(20)], dtype=np.float32).astype(np.float64).reshape(4, 5)
b = np.array([(2 * rng.uniform() - 1) * bound for _ in range(5)], dtype=np.float32).astype(np.float64)
hint = lambda x: np.mean((x.reshape(-1, 4) @ w + b - teacher.reshape(-1, 5)) ** 2)
show("Hint", hint(student), central(hint, student))
weight_grad = central(lambda m: np.mean((student.reshape(-1, 4) @ m + b - teacher.reshape(-1, 5)) ** 2), w)
print("const std::vector<double> kHintWeightGrad = {" + ", ".join(f"{v:.12f}" for v in weight_grad.ravel()) + "};")


def at(x, p):
    q = (x**2 if p == 2 else np.abs(x) ** p).mean(axis=1).reshape(x.shape[0], -1)
    return q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)


s = features((2, 3, 4, 5), 0.9, 0.2)
t = features((2, 6, 4, 5), 0.35, 0.6)
for p in (2, 3):
    loss = lambda x: np.mean((at(x, p) - at(t, p)) ** 2)
    show(f"At{p}", loss(s), central(loss, s))
print("const std::vector<double> kAtMap = {" + ", ".join(f"{v:.12f}" for v in at(s, 2).ravel()[:8]) + "};  // the first 8")
