"""Offline, one-time reference-value generator for tests/representation_geometry_test.cpp (INT-4):
pulsatrix's Anisotropy, MeasureEffectiveRank and TwoNNDimension checked against direct numpy
computations and against the TwoNN code of scikit-dimension and DADApy.

Not run by CTest/CI. Run it with numpy installed and transcribe the printed values into
tests/representation_geometry_test.cpp:

    python3 tools/generate_geometry_reference_values.py

Versions the committed values were generated with: Python 3.12, numpy 2.5.2.

- Anisotropy is the mean cosine similarity over every ordered pair of different rows, computed
  pair by pair (Ethayarajh, EMNLP 2019).
- Effective rank is Roy and Vetterli's (EUSIPCO 2007), from numpy's SVD.
- `twonn_fit` follows scikit-dimension 0.3.7's TwoNN._twonn (skdim/id/_TwoNN.py, BSD-3-Clause):
  mu = r2 / r1, the smallest int(N (1 - discard_fraction)) kept, F = arange(1, kept + 1) / N, and a
  least-squares line through the origin of -log(1 - F) on log(mu). DADApy 0.3.4's
  _compute_id_2NN(algorithm="base") (dadapy/id_estimation.py, Apache 2.0) is the same fit.
- `twonn_mle` follows DADApy's _compute_id_2NN(algorithm="ml"): (N - 1) / sum(log(mu)) over every
  point.

The inputs come from a formula rather than a random generator, so the C++ test builds the same
matrices: x[i][a] = offset + sin(seed + 0.37 i + 1.13 a + 0.05 i a + 0.71 i^2), in float32.
"""

import numpy as np


def make(seed, n, p, offset=0.0):
    i = np.arange(n, dtype=np.float64)[:, None]
    a = np.arange(p, dtype=np.float64)[None, :]
    return (offset + np.sin(seed + 0.37 * i + 1.13 * a + 0.05 * i * a + 0.71 * i * i)).astype(np.float32).astype(np.float64)


def anisotropy(x):
    u = x / np.linalg.norm(x, axis=1, keepdims=True)
    n = len(u)
    total = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                total += float(u[i] @ u[j])
    return total / (n * (n - 1))


def effective_rank(x):
    s = np.linalg.svd(x, compute_uv=False)
    q = s / s.sum()
    q = q[q > 0]
    return float(np.exp(-(q * np.log(q)).sum())), float((s**2).sum() ** 2 / (s**4).sum())


def mus(x):
    d = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    d.sort(axis=1)
    assert (d[:, 0] > 0).all(), "duplicate points"
    return d[:, 1] / d[:, 0]


def twonn_fit(x, discard_fraction=0.1):
    _mu = mus(x)
    N = len(x)
    mu = _mu[np.argsort(_mu)[: int(N * (1 - discard_fraction))]]
    Femp = np.arange(1, 1 + int(N * (1 - discard_fraction))) / N
    lx = np.log(mu).reshape(-1, 1)
    ly = -np.log(1 - Femp).reshape(-1, 1)
    coef, *_ = np.linalg.lstsq(lx, ly, rcond=None)  # LinearRegression(fit_intercept=False)
    return float(coef[0][0])


def twonn_mle(x):
    mu = mus(x)
    return float((len(mu) - 1) / np.sum(np.log(mu)))


def show(name, v):
    print(f"{name}: {v:.17g}")


# Tall, with a common direction: 30 rows of width 12, offset 0.8.
x = make(0.3, 30, 12, offset=0.8)
xc = x - x.mean(0)
show("tall anisotropy", anisotropy(x))
show("tall anisotropy centered", anisotropy(xc))
er, pr = effective_rank(x)
show("tall effective rank", er)
show("tall participation ratio", pr)
er, pr = effective_rank(xc)
show("tall effective rank centered", er)
show("tall participation ratio centered", pr)

# Wide: 10 rows of width 40. Centered, it has rank 9.
w = make(1.1, 10, 40)
wc = w - w.mean(0)
er, pr = effective_rank(w)
show("wide effective rank", er)
show("wide participation ratio", pr)
er, pr = effective_rank(wc)
show("wide effective rank centered", er)
show("wide participation ratio centered", pr)
print("wide centered singular values:", np.linalg.svd(wc, compute_uv=False))

# TwoNN on 200 points of width 3.
t = make(0.7, 200, 3)
show("twonn fit", twonn_fit(t))
show("twonn fit discard 0.25", twonn_fit(t, 0.25))
show("twonn mle", twonn_mle(t))
