"""Offline, one-time reference-value generator for tests/ice_test.cpp (CFS-1): pulsatrix's ICE,
partial dependence and grid construction checked against scikit-learn's partial_dependence.

Not run by CTest/CI. Run it with scikit-learn installed and transcribe the printed C++
initializer lists into tests/ice_test.cpp:

    python3 tools/generate_ice_reference_values.py

Versions the committed values were generated with: Python 3.12, numpy 2.5.3, scikit-learn 1.9.1.

The model is a fixed function, so no training is involved on either side:
    f(x) = sin(x0) * x1 + 0.5 * x2^2 - x0 * x2
It has an interaction between x0 and x1 (ICE curves over x0 fan out with x1) and between x0 and
x2. The background is 30 instances of det(90, 5, 3, 4) reshaped to (30, 3): the same det()
formula as the other generators, float32. det() takes 17 distinct values per feature, so a grid
resolution of 10 exercises the percentile grid and 20 the distinct-values grid.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import partial_dependence


def det(n, mul, add, div):
    ints = np.array([float((i * mul + add) % 17 - 8) for i in range(n)], dtype=np.float32)
    return ints / np.float32(div)


class Fixed(RegressorMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        return np.sin(X[:, 0]) * X[:, 1] + np.float32(0.5) * X[:, 2] ** 2 - X[:, 0] * X[:, 2]


def cpp_float(v):
    text = "%.9g" % (float(v) + 0.0)
    return (text if any(c in text for c in ".e") else text + ".0") + "f"


def emit(name, values):
    print("const std::vector<float> k%s = {" % name)
    line = "   "
    for v in np.asarray(values).reshape(-1).tolist():
        item = " " + cpp_float(v) + ","
        if len(line) + len(item) > 116:
            print(line)
            line = "   "
        line += item
    print(line)
    print("};")


if __name__ == "__main__":
    X = det(90, 5, 3, 4).reshape(30, 3)
    est = Fixed().fit(X)

    r = partial_dependence(est, X, [0], kind="both", grid_resolution=10, method="brute")
    emit("PercentileGrid", r["grid_values"][0])
    emit("Average", r["average"][0])
    emit("Individual", r["individual"][0])

    r = partial_dependence(est, X, [1], kind="average", grid_resolution=20, method="brute")
    emit("DistinctGrid", r["grid_values"][0])

    r = partial_dependence(est, X, [(0, 2)], kind="average", grid_resolution=6, method="brute")
    emit("Grid2DFeature0", r["grid_values"][0])
    emit("Grid2DFeature2", r["grid_values"][1])
    # average[0][i][j]: feature 0 = grid0[i], feature 2 = grid2[j].
    emit("Average2D", r["average"][0])
