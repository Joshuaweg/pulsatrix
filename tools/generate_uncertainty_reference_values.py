"""Offline, one-time reference-value generator for tests/uncertainty_test.cpp (KS-3): pulsatrix's
DecomposeUncertainty, CombineGaussians and GaussianNLLLoss checked against numpy computations of
their definitions.

Not run by CTest/CI. Run it with numpy installed and transcribe the printed values into
tests/uncertainty_test.cpp:

    python3 tools/generate_uncertainty_reference_values.py

Versions the committed values were generated with: Python 3.14.4, numpy 2.5.4.

The samples come from a formula, so the C++ test builds the same values: sample t of input i has
logits 1.8 sin(0.4 + 0.9 i + 1.7 k + 0.6 t (1 + i mod 2) + 0.3 i k), for T = 5 samples, N = 6 inputs
and K = 3 classes, and probabilities their softmax in float64, stored as float32. Odd inputs vary
more between samples than even ones.

  - total = H[mean_t p_t], aleatoric = mean_t H[p_t], epistemic = total - aleatoric, in nats
    (Depeweg et al., ICML 2018, eq. 5);
  - a Gaussian mixture of M members has mean mean_m mu_m and variance mean_m (var_m + mu_m^2) - mu^2
    (Lakshminarayanan et al., NeurIPS 2017, section 2.4);
  - the Gaussian NLL is PyTorch's GaussianNLLLoss(full=False, reduction="mean"),
    mean 0.5 (log var + (y - mu)^2 / var), with var = softplus(s) + 1e-6.
"""

import numpy as np

T, N, K = 5, 6, 3
t = np.arange(T, dtype=np.float64)[:, None, None]
i = np.arange(N, dtype=np.float64)[None, :, None]
k = np.arange(K, dtype=np.float64)[None, None, :]
odd = (np.arange(N) % 2)[None, :, None]
logits = 1.8 * np.sin(0.4 + 0.9 * i + 1.7 * k + 0.6 * t * (1 + odd) + 0.3 * i * k)
e = np.exp(logits - logits.max(axis=2, keepdims=True))
p = (e / e.sum(axis=2, keepdims=True)).astype(np.float32).astype(np.float64)


def entropy(q):
    return -np.sum(np.where(q > 0, q * np.log(np.where(q > 0, q, 1.0)), 0.0), axis=-1)


mean = p.mean(axis=0)
total = entropy(mean)
aleatoric = entropy(p).mean(axis=0)
show = lambda name, v: print(f"{name}: " + ", ".join(repr(float(x)) for x in np.ravel(v)))
show("mean", mean)
show("total", total)
show("aleatoric", aleatoric)
show("epistemic", total - aleatoric)
print("predicted:", ", ".join(str(int(c)) for c in mean.argmax(axis=1)))
show("agreement", (p.argmax(axis=2) == mean.argmax(axis=1)[None, :]).mean(axis=0))

M, n = 4, 5
m = np.arange(M, dtype=np.float64)[:, None]
j = np.arange(n, dtype=np.float64)[None, :]
mu = np.sin(0.5 + 0.7 * j) + 0.2 * np.cos(1.3 * m + j)
var = 0.1 + 0.05 * (1 + np.sin(m + 2 * j))
mix_mu = mu.mean(axis=0)
mix_var = (var + mu ** 2).mean(axis=0) - mix_mu ** 2
show("\nmixture mean", mix_mu)
show("mixture variance", mix_var)
show("mixture aleatoric", var.mean(axis=0))
show("mixture epistemic", mu.var(axis=0))

# GaussianNLLLoss on (N, 2) predictions [mu, s].
x = np.array([[0.3, -1.2], [1.5, 0.4], [-0.7, 2.5], [0.0, -4.0]], dtype=np.float32).astype(np.float64)
y = np.array([0.5, 1.0, -2.0, 0.1], dtype=np.float32).astype(np.float64)
v = np.log1p(np.exp(x[:, 1])) + 1e-6
print("\nnll:", repr(float(np.mean(0.5 * (np.log(v) + (y - x[:, 0]) ** 2 / v)))))
