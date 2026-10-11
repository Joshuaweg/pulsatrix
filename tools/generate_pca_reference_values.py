"""Offline, one-time reference-value generator for tests/pca_test.cpp (INT-5): pulsatrix's FitPca
and PcaProject checked against scikit-learn's PCA.

Not run by CTest/CI. Run it with numpy installed and transcribe the printed values into
tests/pca_test.cpp:

    python3 tools/generate_pca_reference_values.py

Versions the committed values were generated with: Python 3.12, numpy 2.5.2.

`sklearn_pca` follows scikit-learn 1.7.2's PCA._fit_full with svd_solver="full"
(sklearn/decomposition/_pca.py, BSD-3-Clause): center, numpy's SVD, svd_flip with
u_based_decision=False (each component's largest-magnitude entry made positive, the first on a
tie), explained_variance = S^2 / (n - 1), and the ratio over the sum of every component's
variance.

The inputs come from a formula rather than a random generator, so the C++ test builds the same
matrices: x[i][a] = offset + sin(seed + 0.37 i + 1.13 a + 0.05 i a + 0.71 i^2), in float32.
"""

import numpy as np


def make(seed, n, p, offset=0.0):
    i = np.arange(n, dtype=np.float64)[:, None]
    a = np.arange(p, dtype=np.float64)[None, :]
    return (offset + np.sin(seed + 0.37 * i + 1.13 * a + 0.05 * i * a + 0.71 * i * i)).astype(np.float32).astype(np.float64)


def svd_flip_v(vt):
    max_abs_v_rows = np.argmax(np.abs(vt), axis=1)
    signs = np.sign(vt[np.arange(vt.shape[0]), max_abs_v_rows])
    return vt * signs[:, None]


def sklearn_pca(x, k):
    mean = x.mean(0)
    xc = x - mean
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    vt = svd_flip_v(vt)
    ev = s**2 / (len(x) - 1)
    return mean, vt[:k], ev[:k], (ev / ev.sum())[:k]


def show(name, values):
    print(f"{name}: {{" + ", ".join(f"{v:.17g}" for v in np.ravel(values)) + "}")


for label, (seed, n, p) in {"tall": (0.3, 40, 6), "wide": (1.1, 8, 30)}.items():
    x = make(seed, n, p, offset=0.5)
    mean, comps, ev, ratio = sklearn_pca(x, 3)
    # The components must be clearly separated, or their order and signs would be fragile.
    gaps = np.diff(np.linalg.svd(x - mean, compute_uv=False)[:4])
    assert (np.abs(gaps) > 1e-3).all(), "near-tied variances"
    print(f"-- {label}: {n} x {p}")
    show("components", comps)
    show("explained_variance", ev)
    show("explained_variance_ratio", ratio)
    show("projection of the first 3 rows", ((x - mean) @ comps.T)[:3])
