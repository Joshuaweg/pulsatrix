"""Offline, one-time reference-value generator for tests/surrogate_tree_test.cpp (XAI-4): pulsatrix's
FitDecisionTree checked against scikit-learn's DecisionTreeClassifier and DecisionTreeRegressor
(splitter="best").

Not run by CTest/CI. Run it with numpy and scikit-learn installed and transcribe the printed values
into tests/surrogate_tree_test.cpp:

    python3 tools/generate_surrogate_tree_reference_values.py

Versions the committed values were generated with: Python 3.14.4, numpy 2.5.4, scikit-learn 1.9.1.

The inputs come from a formula rather than a random generator, so the C++ test builds the same
data: x[i][a] = sin(seed + 0.37 i + 1.13 a + 0.05 i a + 0.71 i^2), in float32. The class is
(x0 + 0.5 x1 > 0.3) + (x2 - x3 > 0.4), and the regression target x0 x1 + sin(3 x2) + 0.5 x3, both
computed in float64 from the float32 values.

scikit-learn breaks ties between features at random (random_state), pulsatrix by the lower index.
A few samples deep in a classification tree are often separated equally well by several features,
so each tree here is fitted with 20 random states and must come out the same every time: those are
the trees both tie-breaking rules have to agree on. With 80 rows instead of 200, the classifiers
below depend on the random state.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def make(seed, n, p):
    i = np.arange(n, dtype=np.float64)[:, None]
    a = np.arange(p, dtype=np.float64)[None, :]
    return np.sin(seed + 0.37 * i + 1.13 * a + 0.05 * i * a + 0.71 * i * i).astype(np.float32)


def fit(model, x, y, **kwargs):
    trees = [model(random_state=r, **kwargs).fit(x, y) for r in range(20)]
    structures = {(tuple(t.tree_.feature), tuple(t.tree_.n_node_samples)) for t in trees}
    assert len(structures) == 1, f"{model.__name__} {kwargs}: ties change the tree"
    return trees[0]


def show(name, tree):
    t = tree.tree_
    print(f"// {name}: {t.node_count} nodes, depth {tree.get_depth()}")
    print("feature:", ", ".join(str(int(f)) for f in t.feature))
    print("threshold:", ", ".join(repr(float(v)) for v in t.threshold))
    print("samples:", ", ".join(str(int(v)) for v in t.n_node_samples))
    print("impurity:", ", ".join(repr(float(v)) for v in t.impurity))
    if isinstance(tree, DecisionTreeClassifier):
        print("value:", ", ".join(str(int(np.argmax(v[0]))) for v in t.value))
    else:
        print("value:", ", ".join(repr(float(v[0][0])) for v in t.value))
    print("importance:", ", ".join(repr(float(v)) for v in tree.feature_importances_))
    print()


x = make(0.3, 200, 4)
xd = x.astype(np.float64)
cls = ((xd[:, 0] + 0.5 * xd[:, 1] > 0.3).astype(int) + (xd[:, 2] - xd[:, 3] > 0.4).astype(int))
reg = xd[:, 0] * xd[:, 1] + np.sin(3 * xd[:, 2]) + 0.5 * xd[:, 3]
print("classes:", np.bincount(cls))

show("classifier max_depth=3", fit(DecisionTreeClassifier, x, cls, max_depth=3))
show("classifier max_depth=4 min_samples_leaf=5", fit(DecisionTreeClassifier, x, cls, max_depth=4, min_samples_leaf=5))
show("regressor max_depth=3", fit(DecisionTreeRegressor, x, reg, max_depth=3))
