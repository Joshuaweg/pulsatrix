"""Offline, one-time reference-value generator for tests/conformal_test.cpp (KS-3): pulsatrix's split
conformal classifier and regressor checked against MAPIE 1.1.0, and APS against a numpy version of
Angelopoulos and Bates' reference code.

Not run by CTest/CI. Run it with numpy, scikit-learn and MAPIE installed and transcribe the printed
values into tests/conformal_test.cpp:

    python3 tools/generate_conformal_reference_values.py

Versions the committed values were generated with: Python 3.14.4, numpy 2.5.4, scikit-learn 1.9.1,
MAPIE 1.1.0.

The data come from a formula, so the C++ test builds the same values. Example i of a set with
offset o has logits 3 sin(o + 0.53 i + 1.21 k + 0.07 i k + 0.61 i^2) for K = 5 classes, and
probabilities their softmax in float64, stored as float32. Its label is the most probable class,
except where sin(2.3 i + o) < -0.6, where it is the second most probable (ties to the lower index). The calibration
set has 500 examples and o = 0.2, the test set 300 and o = 1.9. For regression, the prediction is
sin(0.8 i + o) and the target that plus 0.4 sin(3.1 i + 2 o) (1 + 0.5 cos(i)).

MAPIE takes the quantile as numpy's "higher" quantile at (n + 1)(1 - alpha) / n; pulsatrix takes
the ceil((n + 1)(1 - alpha))-th smallest score, as Angelopoulos and Bates state it. The two can
differ by one rank; this script checks that they agree at the levels it prints.
"""

import math

import numpy as np
from mapie.classification import SplitConformalClassifier
from mapie.regression import SplitConformalRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

K = 5


def probabilities(n, o):
    i = np.arange(n, dtype=np.float64)[:, None]
    k = np.arange(K, dtype=np.float64)[None, :]
    logits = 3.0 * np.sin(o + 0.53 * i + 1.21 * k + 0.07 * i * k + 0.61 * i * i)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    p = (e / e.sum(axis=1, keepdims=True)).astype(np.float32).astype(np.float64)
    order = np.argsort(-p, axis=1, kind="stable")
    flip = np.sin(2.3 * np.arange(n) + o) < -0.6
    return p, np.where(flip, order[:, 1], order[:, 0])


class Fixed(ClassifierMixin, BaseEstimator):
    """A fitted classifier that returns a stored row of probabilities for each index."""

    def __init__(self, table=None):
        self.table = table

    def fit(self, x, y):
        self.classes_ = np.arange(K)
        return self

    def predict_proba(self, x):
        return self.table[x[:, 0].astype(int)]

    def predict(self, x):
        return self.predict_proba(x).argmax(axis=1)


class FixedRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, table=None):
        self.table = table

    def fit(self, x, y):
        self.fitted_ = True
        return self

    def predict(self, x):
        return self.table[x[:, 0].astype(int)]


MASKS = "0123456789ABCDEFGHIJKLMNOPQRSTUV"


def masks(included):
    """Each set as one character: the bitmask of its classes, class k as bit k."""
    return '"' + "".join(MASKS[int(sum(1 << k for k in np.flatnonzero(row)))] for row in included) + '"'


def rank_quantile(scores, alpha):
    rank = math.ceil((len(scores) + 1) * (1 - alpha) - 1e-9)
    return np.inf if rank > len(scores) else np.sort(scores)[rank - 1]


pc, yc = probabilities(500, 0.2)
pt, yt = probabilities(300, 1.9)
p_all = np.vstack([pc, pt])
x_cal = np.arange(500)[:, None]
x_test = np.arange(500, 800)[:, None]

for alpha in (0.1, 0.05):
    lac = 1 - pc[np.arange(500), yc]
    ours = rank_quantile(lac, alpha)
    mapie = np.quantile(lac, (501) * (1 - alpha) / 500, method="higher")
    assert ours == mapie, (alpha, ours, mapie)
    model = Fixed(p_all).fit(None, None)
    scc = SplitConformalClassifier(estimator=model, confidence_level=1 - alpha, conformity_score="lac", prefit=True)
    scc.conformalize(x_cal, yc)
    _, sets = scc.predict_set(x_test)
    sets = sets[:, :, 0]
    print(f"// LAC, alpha {alpha}: quantile {float(ours)!r}")
    print("sets:", masks(sets))
    print("coverage:", repr(float(sets[np.arange(300), yt].mean())), "mean size:", repr(float(sets.sum(axis=1).mean())))

# APS without randomization, as in Angelopoulos and Bates' notebook, with the rank above.
for alpha in (0.1, 0.05):
    order = np.argsort(-pc, axis=1, kind="stable")
    cum = np.take_along_axis(pc, order, axis=1).cumsum(axis=1)
    scores = np.take_along_axis(cum, np.argsort(order, axis=1), axis=1)[np.arange(500), yc]
    q = rank_quantile(scores, alpha)
    order_t = np.argsort(-pt, axis=1, kind="stable")
    cum_t = np.take_along_axis(pt, order_t, axis=1).cumsum(axis=1)
    included = np.take_along_axis(cum_t <= q, np.argsort(order_t, axis=1), axis=1)
    print(f"// APS, alpha {alpha}: quantile {float(q)!r}")
    print("sets:", masks(included))
    print("coverage:", repr(float(included[np.arange(300), yt].mean())), "mean size:", repr(float(included.sum(axis=1).mean())))

i = np.arange(800, dtype=np.float64)
o = np.where(i < 500, 0.2, 1.9)
j = np.where(i < 500, i, i - 500)
pred = np.sin(0.8 * j + o)
target = pred + 0.4 * np.sin(3.1 * j + 2 * o) * (1 + 0.5 * np.cos(j))
for alpha in (0.1, 0.05):
    reg = FixedRegressor(pred).fit(None, None)
    scr = SplitConformalRegressor(estimator=reg, confidence_level=1 - alpha, conformity_score="absolute", prefit=True)
    scr.conformalize(x_cal, target[:500])
    _, intervals = scr.predict_interval(x_test)
    half = intervals[:, 1, 0] - pred[500:]
    ours = rank_quantile(np.abs(target[:500] - pred[:500]), alpha)
    assert np.allclose(half, ours, rtol=0, atol=1e-12), (alpha, half[:3], ours)
    inside = (intervals[:, 0, 0] <= target[500:]) & (target[500:] <= intervals[:, 1, 0])
    print(f"// absolute residual, alpha {alpha}: quantile {float(ours)!r}, coverage {float(inside.mean())!r}")
