"""Offline, one-time reference-value generator for tests/rsa_svcca_test.cpp (INT-7): pulsatrix's
RSA and SVCCA checked against independent numpy implementations.

Not run by CTest/CI. Run it with numpy installed and transcribe the printed values into
tests/rsa_svcca_test.cpp:

    python3 tools/generate_rsa_svcca_reference_values.py

Versions the committed values were generated with: Python 3.12, numpy 2.5.2.

RSA: `pdist` and `spearmanr` below transcribe the definitions of scipy.spatial.distance's
'correlation', 'cosine' and 'euclidean' metrics (correlation and cosine clipped to [0, 2], as scipy
clips them) and of scipy.stats.spearmanr (Pearson's correlation of ranks, ties given their average
rank, as scipy.stats.rankdata(method='average')). scipy isn't installed where these were run.

SVCCA: Raghu et al.'s recipe written out with numpy's SVD: center, keep the fewest top singular
directions whose cumulative share of the variance exceeds 0.99 (scikit-learn's rule for
PCA(n_components=0.99)), and take the canonical correlations of the kept coordinates. Those come
from QR, as `cca` in the CKA notebook accompanying Kornblith et al. (ICML 2019) computes them:
the singular values of Qx^T Qy. pulsatrix gets them from eigendecompositions instead, so the two
share no code path.

The inputs are made from a formula rather than a random generator, so the C++ test builds the
same matrices: x[i][a] = sin(seed + 0.37 i + 1.13 a + 0.05 i a), in float32.
"""

import numpy as np


def make(seed, n, p):
    i = np.arange(n, dtype=np.float64)[:, None]
    a = np.arange(p, dtype=np.float64)[None, :]
    return np.sin(seed + 0.37 * i + 1.13 * a + 0.05 * i * a).astype(np.float32).astype(np.float64)


def pdist(x, metric):
    n = x.shape[0]
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            u, v = x[i], x[j]
            if metric == "euclidean":
                out.append(np.sqrt(np.sum((u - v) ** 2)))
                continue
            if metric == "correlation":
                u = u - u.mean()
                v = v - v.mean()
            d = 1.0 - np.dot(u, v) / np.sqrt(np.dot(u, u) * np.dot(v, v))
            out.append(np.clip(d, 0.0, 2.0))
    return np.array(out)


def rankdata(a):
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b))


def spearmanr(a, b):
    return pearson(rankdata(a), rankdata(b))


def svcca(x, y, keep=0.99, directions=0):
    def reduce(z):
        z = z - z.mean(0)
        _, s, vt = np.linalg.svd(z, full_matrices=False)
        ratio = np.cumsum(s ** 2) / np.sum(s ** 2)
        k = int(np.searchsorted(ratio, keep, side="right")) + 1
        k = min(directions if directions else k, len(s))
        return z.dot(vt[:k].T), k

    xr, kx = reduce(x)
    yr, ky = reduce(y)
    return canonical_correlations(xr, yr), kx, ky


def canonical_correlations(x, y):
    x = x - x.mean(0)
    y = y - y.mean(0)
    qx, _ = np.linalg.qr(x)
    qy, _ = np.linalg.qr(y)
    return np.linalg.svd(qx.T.dot(qy), compute_uv=False)


def show(label, values):
    print(label + ": {" + ", ".join(f"{v:.17g}" for v in values) + "}")


# RSA: an RDM per metric for 5 inputs, then the RDMs of two 9-input representations compared.
small = make(0.3, 5, 4)
for metric in ("correlation", "euclidean", "cosine"):
    show(f"rdm {metric}", pdist(small, metric))
x = make(0.0, 9, 6)
y = make(2.0, 9, 4)
for metric in ("correlation", "euclidean", "cosine"):
    a, b = pdist(x, metric), pdist(y, metric)
    print(f"rsa {metric} spearman: {spearmanr(a, b):.17g}")
    print(f"rsa {metric} pearson: {pearson(a, b):.17g}")

# Spearman with ties: each tied group gets its average rank.
show("ranks", rankdata(np.array([3.0, 1.0, 3.0, 2.0, 3.0, 1.0])))

# Canonical correlations of the raw representations, their widths adding up to fewer than the
# inputs, as CCA requires.
for name, sx, sy, n, p, q in (("cca narrow", 0.0, 2.0, 30, 6, 4), ("cca equal", 0.5, 1.5, 25, 5, 5)):
    show(name, canonical_correlations(make(sx, n, p), make(sy, n, q)))

def low_rank(seed, n, p, r):
    # r strong directions under weak ones, so the top singular directions are well separated.
    return (make(seed, n, r).dot(make(seed + 5.0, r, p)) + 0.1 * make(seed + 9.0, n, p)).astype(np.float32).astype(np.float64)


def gap_after(z, k):
    s = np.linalg.svd(z - z.mean(0), compute_uv=False)
    return (s[k - 1] - s[k]) / s[0]


# SVCCA, with each side's kept directions: by the variance rule, then 6 of each for
# representations wider than the inputs (through Gram matrices in pulsatrix). The wide ones are
# low rank plus noise: make()'s wide matrices have nearly equal singular values, so their "top"
# directions would be arbitrary.
x = make(0.25, 60, 14)
y = make(1.75, 60, 10)
rho, kx, ky = svcca(x, y)
print(f"svcca: kept {kx} {ky}, mean {rho.mean():.17g}")
show("svcca correlations", rho)
x = low_rank(1.0, 40, 50, 6)
y = low_rank(3.0, 40, 30, 6)
assert gap_after(x, 6) > 1e-2 and gap_after(y, 6) > 1e-2, (gap_after(x, 6), gap_after(y, 6))
rho, kx, ky = svcca(x, y, directions=6)
print(f"svcca wide: kept {kx} {ky}, mean {rho.mean():.17g}")
show("svcca wide correlations", rho)
# The variance rule on the same pair keeps this many: too many for 40 inputs.
print("svcca wide by the rule: kept", svcca(x, y)[1:])
