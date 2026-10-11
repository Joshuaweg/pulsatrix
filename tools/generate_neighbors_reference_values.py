"""Offline, one-time reference-value generator for tests/nearest_neighbors_test.cpp (INT-1):
pulsatrix's NearestNeighbors and MutualKnnAlignment checked against PyTorch and against Huh et
al.'s mutual-kNN alignment.

Not run by CTest/CI. Run it with torch installed and transcribe the printed values into
tests/nearest_neighbors_test.cpp:

    python3 tools/generate_neighbors_reference_values.py

Versions the committed values were generated with: Python 3.12, torch 2.14.1.

`compute_nearest_neighbors` and `mutual_knn` follow platonic-rep's metrics.py (Huh, Cheung, Wang
and Isola, "The Platonic Representation Hypothesis", ICML 2024; MIT license), with features
L2-normalized first as its measure_alignment does.

The inputs come from a formula rather than a random generator, so the C++ test builds the same
matrices: x[i][a] = offset + sin(seed + 0.37 i + 1.13 a + 0.05 i a + 0.71 i²), in float32. The
i² term scrambles the rows, and the offset gives every vector a common component, which centering
removes. The script asserts there are no tied similarities, since argsort orders ties arbitrarily.
"""

import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)


def make(seed, n, p, offset=0.0):
    i = torch.arange(n, dtype=torch.float64)[:, None]
    a = torch.arange(p, dtype=torch.float64)[None, :]
    return (offset + torch.sin(seed + 0.37 * i + 1.13 * a + 0.05 * i * a + 0.71 * i * i)).float().double()


def compute_nearest_neighbors(feats, topk=1):
    assert feats.ndim == 2, f"Expected feats to be 2D, got {feats.ndim}"
    knn = (feats @ feats.T).fill_diagonal_(-1e8).argsort(dim=1, descending=True)[:, :topk]
    return knn


def mutual_knn(feats_A, feats_B, topk):
    knn_A = compute_nearest_neighbors(feats_A, topk)
    knn_B = compute_nearest_neighbors(feats_B, topk)
    n = knn_A.shape[0]
    range_tensor = torch.arange(n).unsqueeze(1)
    lvm_mask = torch.zeros(n, n)
    llm_mask = torch.zeros(n, n)
    lvm_mask[range_tensor, knn_A] = 1.0
    llm_mask[range_tensor, knn_B] = 1.0
    acc = (lvm_mask * llm_mask).sum(dim=1) * 1.0 / topk
    return acc.mean().item()


def assert_no_ties(scores):
    s = scores.clone()
    s.fill_diagonal_(float("nan"))
    for row in s:
        vals = row[~torch.isnan(row)]
        assert len(torch.unique(vals)) == len(vals), "tied similarities"


def show(name, idx):
    print(f"{name}: {{" + ", ".join(str(int(v)) for v in idx.flatten()) + "}")


x = make(0.0, 12, 5, offset=0.8)
xn = F.normalize(x, dim=-1)
assert_no_ties(xn @ xn.T)
show("cosine k=3", compute_nearest_neighbors(xn, 3))

xc = x - x.mean(0, keepdim=True)
xcn = F.normalize(xc, dim=-1)
assert_no_ties(xcn @ xcn.T)
show("centered cosine k=3", compute_nearest_neighbors(xcn, 3))

d = torch.cdist(x, x)
assert_no_ties(d)
d.fill_diagonal_(float("inf"))
show("euclidean k=3", d.argsort(dim=1)[:, :3])
print(f"euclidean nearest distances: {{{', '.join(f'{v:.17g}' for v in d.sort(dim=1).values[:, 0])}}}")

# Alignment between two representations of the same 40 inputs, of different widths.
a = make(0.5, 40, 16)
b = make(2.5, 40, 24)
for k in (1, 5, 10):
    an = F.normalize(a, dim=-1)
    bn = F.normalize(b, dim=-1)
    assert_no_ties(an @ an.T)
    assert_no_ties(bn @ bn.T)
    print(f"mutual_knn k={k}: {mutual_knn(an, bn, k):.17g}")
