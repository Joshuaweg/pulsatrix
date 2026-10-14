"""Offline, one-time reference-value generator for tests/soft_decision_tree_test.cpp (KD-3):
pulsatrix's SoftDecisionTree checked against numpy, in float64.

Not run by CTest/CI. Run it with numpy installed and transcribe the printed values into
tests/soft_decision_tree_test.cpp:

    python3 tools/generate_soft_decision_tree_reference_values.py

Versions the committed values were generated with: Python 3.14.4, numpy 2.5.4.

Frosst and Hinton (2017), depth 2 (3 inner nodes, 4 leaves), 3 inputs, 3 classes, beta 1.5,
penalty 0.1, five inputs:

  - gate i: p_i = sigmoid(beta (x . W[:, i] + b_i)), the probability of going right; node i's
    children are 2i + 1 (left) and 2i + 2 (right);
  - W (3, 3) and b drawn as pulsatrix draws them: PortableRng(seed 2), uniform in +-1/sqrt(3),
    the weights row-major and then the bias, rounded to float32;
  - leaf logits phi[l][k] = float32(0.8 sin(1.1 (3 l + k) + 0.4)), Q_l = softmax(phi_l);
  - output y = log sum_l P_l Q_l;
  - penalty C = -lambda sum_i 2^-depth(i) 0.5 (log alpha_i + log(1 - alpha_i)), alpha_i the
    reach-weighted mean of p_i over the batch.

The test's loss is sum(G * y) + C with G[n][k] = float32(sin(0.6 (3 n + k) + 0.9)); its gradients
with respect to the inputs, W, b and phi are central differences.
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


def f32(a):
    return np.asarray(a, dtype=np.float32).astype(np.float64)


DEPTH, INNER, LEAVES, D, C, N, BETA, LAMBDA = 2, 3, 4, 3, 3, 5, 1.5, 0.1
rng = PortableRng(2)
bound = 1 / np.sqrt(D)
W = f32([(2 * rng.uniform() - 1) * bound for _ in range(D * INNER)]).reshape(D, INNER)
B = f32([(2 * rng.uniform() - 1) * bound for _ in range(INNER)])
PHI = f32([0.8 * np.sin(1.1 * k + 0.4) for k in range(LEAVES * C)]).reshape(LEAVES, C)
X = f32([np.sin(0.37 * k + 0.5 * (k % 3)) for k in range(N * D)]).reshape(N, D)
G = f32([np.sin(0.6 * k + 0.9) for k in range(N * C)]).reshape(N, C)


def forward(x, w, b, phi):
    p = 1 / (1 + np.exp(-BETA * (x @ w + b)))
    reach = np.zeros((len(x), INNER + LEAVES))
    reach[:, 0] = 1
    for i in range(INNER):
        reach[:, 2 * i + 1] = reach[:, i] * (1 - p[:, i])
        reach[:, 2 * i + 2] = reach[:, i] * p[:, i]
    q = np.exp(phi - phi.max(axis=1, keepdims=True))
    q /= q.sum(axis=1, keepdims=True)
    y = np.log(reach[:, INNER:] @ q)
    penalty = 0.0
    for i in range(INNER):
        alpha = (reach[:, i] * p[:, i]).sum() / reach[:, i].sum()
        depth = int(np.floor(np.log2(i + 1)))
        penalty -= LAMBDA * 2.0**-depth * 0.5 * (np.log(alpha) + np.log(1 - alpha))
    return y, penalty, reach[:, INNER:]


def loss(x, w, b, phi):
    y, penalty, _ = forward(x, w, b, phi)
    return (G * y).sum() + penalty


def central(f, a, h=1e-6):
    g = np.zeros_like(a)
    for i in np.ndindex(a.shape):
        up, down = a.copy(), a.copy()
        up[i] += h
        down[i] -= h
        g[i] = (f(up) - f(down)) / (2 * h)
    return g


def show(name, values):
    print(f"const std::vector<double> k{name} = {{" + ", ".join(f"{v:.12f}" for v in np.ravel(values)) + "};")


y, penalty, leaf = forward(X, W, B, PHI)
show("Output", y)
print(f"const double kPenalty = {penalty:.12f};")
show("LeafReach", leaf)
show("GradInput", central(lambda a: loss(a, W, B, PHI), X))
show("GradWeight", central(lambda a: loss(X, a, B, PHI), W))
show("GradBias", central(lambda a: loss(X, W, a, PHI), B))
show("GradLeaves", central(lambda a: loss(X, W, B, a), PHI))
