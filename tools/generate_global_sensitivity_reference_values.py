"""Offline, one-time reference-value generator for tests/global_sensitivity_test.cpp (CFS-4):
pulsatrix's Morris and Sobol analyzers checked against SALib's on the same samples and outputs.

Not run by CTest/CI. Run it with SALib and numpy installed and transcribe the printed C++
initializer lists into tests/global_sensitivity_test.cpp:

    python3 tools/generate_global_sensitivity_reference_values.py

Versions the committed values were generated with: Python 3.12, numpy 2.5.3, SALib 1.6.0.

Model: the Ishigami function, y = sin(x1) + 7 sin(x2)^2 + 0.1 x3^4 sin(x1), each x on [-pi, pi],
evaluated in float32 so both sides analyze the same outputs. The samples come from SALib's own
samplers (seed 1); the C++ test feeds them to AnalyzeMorris and AnalyzeSobol. The bootstrap
confidence intervals use different generators on the two sides, so only their size is compared.
"""

import numpy as np
from SALib.analyze import morris as morris_analyze
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import morris as morris_sample
from SALib.sample import sobol as sobol_sample

PROBLEM = {"num_vars": 3, "names": ["x1", "x2", "x3"], "bounds": [[-np.pi, np.pi]] * 3}


def ishigami(X):
    X = np.asarray(X, dtype=np.float32)
    s1 = np.sin(X[:, 0])
    return (s1 + np.float32(7) * np.sin(X[:, 1]) ** 2 + np.float32(0.1) * X[:, 2] ** 4 * s1).astype(np.float32)


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
    X = morris_sample.sample(PROBLEM, N=12, num_levels=4, seed=1).astype(np.float32)
    Y = ishigami(X)
    m = morris_analyze.analyze(PROBLEM, X.astype(np.float64), Y.astype(np.float64), num_levels=4,
                               num_resamples=1000, seed=1)
    emit("MorrisSamples", X)
    emit("MorrisOutputs", Y)
    emit("MorrisMu", m["mu"])
    emit("MorrisMuStar", m["mu_star"])
    emit("MorrisSigma", m["sigma"])
    emit("MorrisMuStarConf", m["mu_star_conf"])

    Xs = sobol_sample.sample(PROBLEM, 64, calc_second_order=False, seed=1)
    Ys = ishigami(Xs)
    s = sobol_analyze.analyze(PROBLEM, Ys.astype(np.float64), calc_second_order=False, num_resamples=200, seed=1)
    emit("SobolOutputs", Ys)
    emit("SobolS1", s["S1"])
    emit("SobolST", s["ST"])
    emit("SobolS1Conf", s["S1_conf"])
    emit("SobolSTConf", s["ST_conf"])
