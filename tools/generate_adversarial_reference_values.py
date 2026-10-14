"""Offline, one-time reference-value generator for tests/adversarial_test.cpp (KS-4): pulsatrix's
Fgsm and Pgd checked against a numpy implementation of torchattacks' FGSM, PGD and PGDL2 updates.

Not run by CTest/CI. Run it with numpy installed and transcribe the printed values into
tests/adversarial_test.cpp:

    python3 tools/generate_adversarial_reference_values.py

Versions the committed values were generated with: Python 3.14.4, numpy 2.5.4.

The network and inputs come from formulas, so the C++ test builds the same ones: a
Linear(4, 8) -> ReLU -> Linear(8, 3) classifier with

    W1[i, j] = 0.7 sin(1.3 i + 0.7 j + 0.2),  b1[j] = 0.1 cos(j)
    W2[j, k] = 0.6 cos(0.9 j + 1.1 k + 0.4),  b2[k] = 0.05 k

(weights stored (in, out), as LinearModule does), six inputs x[n, i] = 0.5 + 0.4 sin(0.8 n + 1.9 i + 0.3)
and labels n mod 3. Every attack clips into [0, 1] and starts at the input (no random start), so it
is deterministic. The updates, as torchattacks writes them (v3.5):

  - FGSM:  x' = clip(x + eps sign(g))
  - FGM (L2): x' = clip(x + eps g / (|g| + 1e-10))
  - PGD:   x <- x + alpha sign(g); delta = clamp(x - x0, -eps, eps); x <- clip(x0 + delta)
  - PGDL2: x <- x + alpha g / (|g| + 1e-10); delta = x - x0; delta *= min(1, eps / |delta|);
           x <- clip(x0 + delta)

where g is the gradient of the cross-entropy of the label with respect to the input, and for a
targeted attack the step goes against the gradient of the target's cross-entropy.
"""

import numpy as np

i = np.arange(4)[:, None]
j = np.arange(8)[None, :]
W1 = (0.7 * np.sin(1.3 * i + 0.7 * j + 0.2)).astype(np.float32).astype(np.float64)
b1 = (0.1 * np.cos(np.arange(8))).astype(np.float32).astype(np.float64)
j2 = np.arange(8)[:, None]
k2 = np.arange(3)[None, :]
W2 = (0.6 * np.cos(0.9 * j2 + 1.1 * k2 + 0.4)).astype(np.float32).astype(np.float64)
b2 = (0.05 * np.arange(3)).astype(np.float32).astype(np.float64)
n = np.arange(6)[:, None]
x0 = (0.5 + 0.4 * np.sin(0.8 * n + 1.9 * np.arange(4)[None, :] + 0.3)).astype(np.float32).astype(np.float64)
labels = np.arange(6) % 3


def forward(x):
    h = np.maximum(x @ W1 + b1, 0.0)
    return h, h @ W2 + b2


def loss_and_grad(x, y):
    h, z = forward(x)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    loss = -np.log(p[np.arange(len(y)), y])
    dz = p.copy()
    dz[np.arange(len(y)), y] -= 1.0
    dh = (dz @ W2.T) * (h > 0)
    return loss, dh @ W1.T


def predict(x):
    return forward(x)[1].argmax(axis=1)


def l2(v):
    return np.sqrt((v * v).sum(axis=1, keepdims=True))


def fgsm(eps, norm, y, targeted=False):
    _, g = loss_and_grad(x0, y)
    d = -1.0 if targeted else 1.0
    step = np.sign(g) if norm == "linf" else g / (l2(g) + 1e-10)
    return np.clip(x0 + d * eps * step, 0.0, 1.0)


def pgd(eps, alpha, steps, norm, y, targeted=False):
    x = x0.copy()
    d = -1.0 if targeted else 1.0
    for _ in range(steps):
        _, g = loss_and_grad(x, y)
        x = x + d * alpha * (np.sign(g) if norm == "linf" else g / (l2(g) + 1e-10))
        delta = x - x0
        if norm == "linf":
            delta = np.clip(delta, -eps, eps)
        else:
            delta = delta * np.minimum(1.0, eps / l2(delta))
        x = np.clip(x0 + delta, 0.0, 1.0)
    return x


def show(name, x, y):
    loss, _ = loss_and_grad(x, y)
    print(f"// {name}")
    print("const std::vector<float> k" + name + " = {")
    for row in x:
        print("    " + ", ".join(f"{v:.8f}f" for v in row) + ",")
    print("};")
    print(f"const std::vector<int64_t> k{name}Predicted = {{{', '.join(str(v) for v in predict(x))}}};")
    print(f"const std::vector<double> k{name}Loss = {{{', '.join(f'{v:.10f}' for v in loss)}}};")
    print()


clean_loss, _ = loss_and_grad(x0, labels)
print(f"const std::vector<int64_t> kCleanPredicted = {{{', '.join(str(v) for v in predict(x0))}}};")
print(f"const std::vector<double> kCleanLoss = {{{', '.join(f'{v:.10f}' for v in clean_loss)}}};")
print()
targets = (np.arange(6) + 1) % 3
show("FgsmLinf", fgsm(0.1, "linf", labels), labels)
show("FgmL2", fgsm(0.3, "l2", labels), labels)
show("PgdLinf", pgd(0.1, 0.025, 10, "linf", labels), labels)
show("PgdL2", pgd(0.5, 0.1, 10, "l2", labels), labels)
show("PgdTargeted", pgd(0.2, 0.02, 20, "linf", targets, targeted=True), targets)
