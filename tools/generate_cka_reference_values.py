"""Offline, one-time reference-value generator for tests/representation_similarity_test.cpp
(INT-2): pulsatrix's LinearCka checked against Kornblith et al.'s reference implementation.

Not run by CTest/CI. Run it with numpy installed and transcribe the printed values into
tests/representation_similarity_test.cpp:

    python3 tools/generate_cka_reference_values.py

Versions the committed values were generated with: Python 3.12, numpy 2.5.2.

`gram_linear`, `center_gram`, `cka` and `feature_space_linear_cka` below follow the CKA notebook
that accompanies Kornblith, Norouzi, Lee and Hinton, "Similarity of Neural Network
Representations Revisited", ICML 2019 (google-research/google-research, representation_similarity,
Apache 2.0). The two ways must agree, which this script asserts.

The inputs are made from a formula rather than a random generator, so the C++ test builds the
same matrices: x[i][a] = sin(seed + 0.37 i + 1.13 a + 0.05 i a), in float32.
"""

import numpy as np


def gram_linear(x):
    return x.dot(x.T)


def center_gram(gram, unbiased=False):
    if not np.allclose(gram, gram.T):
        raise ValueError("Input must be a symmetric matrix.")
    gram = gram.copy()
    if unbiased:
        n = gram.shape[0]
        np.fill_diagonal(gram, 0)
        means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
        means -= np.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        np.fill_diagonal(gram, 0)
    else:
        means = np.mean(gram, 0, dtype=np.float64)
        means -= np.mean(means) / 2
        gram -= means[:, None]
        gram -= means[None, :]
    return gram


def cka(gram_x, gram_y, debiased=False):
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)
    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())
    return scaled_hsic / (np.linalg.norm(gram_x) * np.linalg.norm(gram_y))


def _debiased_dot_product_similarity_helper(xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x,
                                            squared_norm_y, n):
    return (xty - n / (n - 2.0) * sum_squared_rows_x.dot(sum_squared_rows_y)
            + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))


def feature_space_linear_cka(features_x, features_y, debiased=False):
    features_x = features_x - np.mean(features_x, 0, keepdims=True)
    features_y = features_y - np.mean(features_y, 0, keepdims=True)
    dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
    normalization_x = np.linalg.norm(features_x.T.dot(features_x))
    normalization_y = np.linalg.norm(features_y.T.dot(features_y))
    if debiased:
        n = features_x.shape[0]
        sum_squared_rows_x = np.einsum("ij,ij->i", features_x, features_x)
        sum_squared_rows_y = np.einsum("ij,ij->i", features_y, features_y)
        squared_norm_x = np.sum(sum_squared_rows_x)
        squared_norm_y = np.sum(sum_squared_rows_y)
        dot_product_similarity = _debiased_dot_product_similarity_helper(
            dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y, n)
        normalization_x = np.sqrt(_debiased_dot_product_similarity_helper(
            normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x, squared_norm_x, squared_norm_x, n))
        normalization_y = np.sqrt(_debiased_dot_product_similarity_helper(
            normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y, squared_norm_y, squared_norm_y, n))
    return dot_product_similarity / (normalization_x * normalization_y)


def make(seed, n, p):
    i = np.arange(n, dtype=np.float64)[:, None]
    a = np.arange(p, dtype=np.float64)[None, :]
    return np.sin(seed + 0.37 * i + 1.13 * a + 0.05 * i * a).astype(np.float32).astype(np.float64)


# (name, seed_x, seed_y, n, p, q): more inputs than features, more features than inputs, and a
# pair where y is a function of x.
CASES = [
    ("narrow", 0.0, 2.0, 10, 4, 3),
    ("wide", 0.5, 1.5, 6, 20, 15),
    ("square", 1.0, 3.0, 12, 12, 12),
]

for name, sx, sy, n, p, q in CASES:
    x = make(sx, n, p)
    y = make(sy, n, q)
    for debiased in (False, True):
        g = cka(gram_linear(x), gram_linear(y), debiased=debiased)
        f = feature_space_linear_cka(x, y, debiased=debiased)
        assert abs(g - f) < 1e-12, (name, debiased, g, f)
        print(f"{name} {'debiased' if debiased else 'biased'}: {g:.17g}")

# y = tanh(x W) for a fixed W: related but not linearly.
x = make(0.25, 16, 6)
w = make(4.0, 6, 5)
y = np.tanh(x.dot(w)).astype(np.float32).astype(np.float64)
for debiased in (False, True):
    g = cka(gram_linear(x), gram_linear(y), debiased=debiased)
    assert abs(g - feature_space_linear_cka(x, y, debiased=debiased)) < 1e-12
    print(f"tanh {'debiased' if debiased else 'biased'}: {g:.17g}")
