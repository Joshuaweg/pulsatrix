"""Offline, one-time reference-value generator for tests/distillation_loss_test.cpp (KD-1):
pulsatrix's DistillationLoss checked against numpy, in float64.

Not run by CTest/CI. Run it with numpy installed and transcribe the printed values into
tests/distillation_loss_test.cpp:

    python3 tools/generate_distillation_reference_values.py

Versions the committed values were generated with: Python 3.14.4, numpy 2.5.4.

The soft term is written the way PyTorch computes it,
F.kl_div(F.log_softmax(s / T, -1), F.softmax(t / T, -1), reduction="sum") * T * T, over the kept
rows (reverse KL swaps the two distributions), divided by the number of kept rows. The hard term is
F.cross_entropy(s, y, ignore_index=-100) at T = 1. The gradients are central differences of the
loss, not the closed form, so they check pulsatrix's derivation as well as its arithmetic.

The logits are float32(2 sin(0.7 k + 0.3)) for the student and float32(3 cos(0.45 k + 0.1)) for the
teacher, k the flat index.
"""

import numpy as np

IGNORE = -100


def logits(shape):
    k = np.arange(int(np.prod(shape)), dtype=np.float64)
    s = (2 * np.sin(0.7 * k + 0.3)).astype(np.float32).astype(np.float64).reshape(shape)
    t = (3 * np.cos(0.45 * k + 0.1)).astype(np.float32).astype(np.float64).reshape(shape)
    return s, t


def log_softmax(z):
    m = z.max(axis=-1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=-1, keepdims=True))


def parts(s, t, targets, temperature, reverse):
    c = s.shape[-1]
    s2, t2 = s.reshape(-1, c), t.reshape(-1, c)
    keep = np.ones(len(s2), bool) if targets is None else targets.reshape(-1) != IGNORE
    ls, lt = log_softmax(s2[keep] / temperature), log_softmax(t2[keep] / temperature)
    if reverse:
        ls, lt = lt, ls
    n = keep.sum()
    soft = (np.exp(lt) * (lt - ls)).sum() * temperature**2 / n
    hard = 0.0
    if targets is not None:
        y = targets.reshape(-1)[keep].astype(int)
        hard = -log_softmax(s2[keep])[np.arange(n), y].sum() / n
    return soft, hard


def loss(s, t, targets, temperature, alpha, reverse):
    soft, hard = parts(s, t, targets, temperature, reverse)
    return soft if targets is None else alpha * soft + (1 - alpha) * hard


def gradient(s, t, targets, temperature, alpha, reverse, h=1e-6):
    g = np.zeros_like(s)
    for i in np.ndindex(s.shape):
        up, down = s.copy(), s.copy()
        up[i] += h
        down[i] -= h
        g[i] = (loss(up, t, targets, temperature, alpha, reverse) - loss(down, t, targets, temperature, alpha, reverse)) / (2 * h)
    return g


def show(name, shape, targets, temperature, alpha, reverse):
    s, t = logits(shape)
    soft, hard = parts(s, t, targets, temperature, reverse)
    total = loss(s, t, targets, temperature, alpha, reverse)
    g = gradient(s, t, targets, temperature, alpha, reverse)
    print(f"// {name}: shape {shape}, T {temperature}, alpha {alpha}, {'reverse' if reverse else 'forward'}")
    print(f"const double k{name}Loss = {total:.10f}, k{name}Soft = {soft:.10f}, k{name}Hard = {hard:.10f};")
    print(f"const std::vector<double> k{name}Grad = {{" + ", ".join(f"{v:.10f}" for v in g.reshape(-1)) + "};")


show("Forward", (4, 5), None, 2.0, 0.5, False)
show("Reverse", (2, 3, 6), None, 3.0, 0.5, True)
targets = np.array([[1, IGNORE, 3, 0], [IGNORE, 2, 4, 1]], dtype=np.float64)
show("ForwardTargets", (2, 4, 5), targets, 4.0, 0.3, False)
show("ReverseTargets", (2, 4, 5), targets, 1.5, 0.7, True)
