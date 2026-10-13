"""Offline, one-time reference-value generator for tests/performance_test.cpp (KS-3): pulsatrix's
EvaluateClassifier, EvaluateRegressor and PerSliceError checked against scikit-learn's metrics and
SciPy's Fisher exact test.

Not run by CTest/CI. Run it with numpy, SciPy and scikit-learn installed and transcribe the printed
values into tests/performance_test.cpp:

    python3 tools/generate_performance_reference_values.py

Versions the committed values were generated with: Python 3.14.4, numpy 2.5.4, SciPy 1.18.1,
scikit-learn 1.9.1.

The data come from a formula rather than a random generator, so the C++ test builds the same values:
  - logits[i][k] = 2.5 sin(0.3 + 0.37 i + 1.13 k + 0.05 i k + 0.71 i^2), for 200 inputs and 4 classes;
    the probabilities are their softmax, computed in float64 and stored as float32;
  - the label is the most probable class, except where sin(1.7 i) is below a threshold (0.3 for
    slice "c", -0.5 otherwise), where it is (that class + 1 + i mod 3) mod 4;
  - input i is in slice "a", "b" or "c" for i mod 3 = 0, 1, 2, so slice "c" is wrong more often;
  - for regression, the prediction is sin(0.9 i) and the target that plus 0.3 cos(2.1 i).

scikit-learn has no expected calibration error, so ECE is computed here from its definition (Guo,
Pleiss, Sun and Weinberger, ICML 2017): equal-width bins ((m-1)/M, m/M] of the top-label confidence,
ECE = sum_m |B_m|/n |accuracy(B_m) - confidence(B_m)|. The Wilson interval is computed from its
formula at z = 1.959963984540054.
"""

import warnings

import numpy as np
from scipy.stats import fisher_exact
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, log_loss, mean_absolute_error,
                             mean_squared_error, precision_recall_fscore_support, r2_score)

# The float32 probabilities sum to 1 only to float precision; scikit-learn warns and uses them as they are.
warnings.filterwarnings("ignore", message="The y_prob values do not sum to one")

N, K = 200, 4
i = np.arange(N, dtype=np.float64)
k = np.arange(K, dtype=np.float64)
logits = 2.5 * np.sin(0.3 + 0.37 * i[:, None] + 1.13 * k[None, :] + 0.05 * i[:, None] * k[None, :] + 0.71 * i[:, None] ** 2)
e = np.exp(logits - logits.max(axis=1, keepdims=True))
probs = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
p = probs.astype(np.float64)
top = logits.argmax(axis=1)
slice_of = np.array(["abc"[n % 3] for n in range(N)])
threshold = np.where(slice_of == "c", 0.3, -0.5)
flip = np.sin(1.7 * i) < threshold
labels = np.where(flip, (top + 1 + np.arange(N) % 3) % K, top)
pred = p.argmax(axis=1)


def show(name, values):
    print(f"{name}: " + ", ".join(repr(float(v)) for v in np.ravel(values)))


def ece(confidence, correct, bins):
    edges = np.linspace(0.0, 1.0, bins + 1)
    total, worst, rows = 0.0, 0.0, []
    for m in range(bins):
        inside = (confidence > edges[m]) & (confidence <= edges[m + 1])
        if m == 0:
            inside |= confidence == 0.0
        count = int(inside.sum())
        if count == 0:
            rows.append((count, 0.0, 0.0))
            continue
        acc, conf = correct[inside].mean(), confidence[inside].mean()
        total += count / len(confidence) * abs(acc - conf)
        worst = max(worst, abs(acc - conf))
        rows.append((count, conf, acc))
    return total, worst, rows


print("// classification")
print("accuracy:", repr(accuracy_score(labels, pred)))
print("log_loss:", repr(log_loss(labels, p, labels=list(range(K)))))
print("brier:", repr(brier_score_loss(labels, p, labels=list(range(K)), scale_by_half=False)))
precision, recall, f1, support = precision_recall_fscore_support(labels, pred, labels=list(range(K)), zero_division=0)
show("precision", precision)
show("recall", recall)
show("f1", f1)
print("support:", ", ".join(str(int(v)) for v in support))
print("macro_f1:", repr(float(f1.mean())))
print("confusion:", ", ".join(str(int(v)) for v in confusion_matrix(labels, pred, labels=list(range(K))).ravel()))
for bins in (15, 10):
    total, worst, rows = ece(p.max(axis=1), (pred == labels).astype(np.float64), bins)
    print(f"ece{bins}:", repr(float(total)), "mce:", repr(float(worst)))
    print(f"bins{bins} count:", ", ".join(str(r[0]) for r in rows))
    show(f"bins{bins} confidence", [r[1] for r in rows])
    show(f"bins{bins} accuracy", [r[2] for r in rows])

print("\n// per-slice error, one-sided Fisher exact test of each slice against the rest")
wrong = pred != labels
z = 1.959963984540054
for s in "abc":
    inside = slice_of == s
    a, b = int(wrong[inside].sum()), int((~wrong[inside]).sum())
    c, d = int(wrong[~inside].sum()), int((~wrong[~inside]).sum())
    n = a + b
    rate = a / n
    centre = (rate + z * z / (2 * n)) / (1 + z * z / n)
    half = z / (1 + z * z / n) * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n))
    _, pv = fisher_exact([[a, b], [c, d]], alternative="greater")
    print(f'{{"{s}", {n}, {a}, {rate!r}, {float(centre - half)!r}, {float(centre + half)!r}, {c / (c + d)!r}, {float(pv)!r}}},')
print("overall error:", repr(float(wrong.mean())))

print("\n// regression")
yp = np.sin(0.9 * i)
yt = yp + 0.3 * np.cos(2.1 * i)
print("mse:", repr(mean_squared_error(yt, yp)))
print("mae:", repr(mean_absolute_error(yt, yp)))
print("r2:", repr(r2_score(yt, yp)))
