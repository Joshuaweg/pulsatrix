"""Offline, one-time reference-value generator for tests/explanation_agreement_test.cpp (KD-2):
pulsatrix's CompareExplanations checked against scipy and numpy.

Not run by CTest/CI. Run it with numpy and scipy installed and transcribe the printed values into
tests/explanation_agreement_test.cpp:

    python3 tools/generate_explanation_agreement_reference_values.py

Versions the committed values were generated with: Python 3.14.4, numpy 2.5.4, scipy (as installed
with scikit-learn 1.9.1).

Six inputs, twelve features. The teacher's attribution of input i, feature j, is
float32(sin(0.9 (12 i + j) + 0.2)); the student's is float32(0.8 teacher + 0.5 cos(1.7 (12 i + j))),
rounded to one decimal so some values tie. Magnitudes are compared, as by default:

  - Spearman: scipy.stats.spearmanr (ties get their average rank);
  - top-k overlap: k = round(0.25 * 12) = 3, the k largest by a stable descending argsort (ties to
    the lower index);
  - cosine: numpy;
  - CKA: Kornblith et al.'s debiased linear CKA, from tools/generate_cka_reference_values.py;
  - the permutation test: the mean Spearman of the teacher's map of input i with the student's of
    input label[i], against 199 relabelings drawn the way pulsatrix draws them (PortableRng,
    SplitMix64; a Fisher-Yates shuffle with index next() % i, each shuffle applied to the previous
    one). The p-value counts the real labeling.
"""

import numpy as np
from scipy.stats import spearmanr

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


def center_gram(gram):
    n = gram.shape[0]
    gram = gram.copy()
    np.fill_diagonal(gram, 0)
    means = np.sum(gram, 0) / (n - 2)
    means -= np.sum(means) / (2 * (n - 1))
    gram -= means[:, None]
    gram -= means[None, :]
    np.fill_diagonal(gram, 0)
    return gram


def cka_debiased(x, y):
    gx, gy = center_gram(x @ x.T), center_gram(y @ y.T)
    return gx.ravel() @ gy.ravel() / (np.linalg.norm(gx) * np.linalg.norm(gy))


n, f = 6, 12
idx = np.arange(n * f, dtype=np.float64).reshape(n, f)
teacher = np.sin(0.9 * idx + 0.2).astype(np.float32)
student = np.round((0.8 * teacher.astype(np.float64) + 0.5 * np.cos(1.7 * idx)), 1).astype(np.float32)
t, s = np.abs(teacher.astype(np.float64)), np.abs(student.astype(np.float64))

k = max(1, round(0.25 * f))
rho = np.array([spearmanr(t[i], s[i]).statistic for i in range(n)])
top = np.array([len(set(np.argsort(-t[i], kind="stable")[:k]) & set(np.argsort(-s[i], kind="stable")[:k])) / k for i in range(n)])
cos = np.array([t[i] @ s[i] / np.sqrt((t[i] @ t[i]) * (s[i] @ s[i])) for i in range(n)])
c = np.array([[spearmanr(t[i], s[j]).statistic for j in range(n)] for i in range(n)])
mismatched = (c.sum() - np.trace(c)) / (n * (n - 1))

rng = PortableRng(5)
label = list(range(n))
observed = np.mean([c[i, label[i]] for i in range(n)])
at_least = 1
for _ in range(199):
    for i in range(n, 1, -1):
        j = rng.next() % i
        label[i - 1], label[j] = label[j], label[i - 1]
    if np.mean([c[i, label[i]] for i in range(n)]) >= observed - 1e-12:
        at_least += 1

def lit(v):
    text = f"{v:.9g}"
    return (text if "." in text or "e" in text else text + ".0") + "f"


print("// teacher, student (float32 values):")
print("const std::vector<float> kTeacher = {" + ", ".join(lit(v) for v in teacher.ravel()) + "};")
print("const std::vector<float> kStudent = {" + ", ".join(lit(v) for v in student.ravel()) + "};")
print("const std::vector<double> kSpearman = {" + ", ".join(f"{v:.12f}" for v in rho) + "};")
print("const std::vector<double> kTopK = {" + ", ".join(f"{v:.12f}" for v in top) + "};")
print("const std::vector<double> kCosine = {" + ", ".join(f"{v:.12f}" for v in cos) + "};")
print(f"const double kCka = {cka_debiased(t, s):.12f};")
print(f"const double kMismatched = {mismatched:.12f};")
print(f"const double kPValue = {at_least / 200:.12f};  // {at_least} of 200")
