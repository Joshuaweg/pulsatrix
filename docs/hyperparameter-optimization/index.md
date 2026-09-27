# Hyperparameter Optimization

A from-scratch, typed hyperparameter search space plus every major HPO family, all
interoperable through the same `SearchSpace`/`Configuration`/`Trial` core: grid and random
search, Bayesian optimization (Gaussian-process and Tree-structured Parzen Estimator
surrogates), and bandit-based early-stopping (Successive Halving, Hyperband, ASHA). Every
algorithm here is a from-scratch reimplementation, never a runtime dependency on scikit-learn,
Optuna, or any other Python HPO library.

## What's inside

- **Core**: `SearchSpace` (Continuous/LogUniform/Integer/Categorical parameters),
  `Configuration` (a concrete parameter assignment), `Trial` (a configuration plus its metric
  history), `RandomSample`/`GridSample`.
- **Bayesian optimization**: `GaussianProcessRegressor` (squared-exponential kernel,
  posterior mean/variance), `ExpectedImprovement`/`ProbabilityOfImprovement`/
  `UpperConfidenceBound` acquisition functions, `RunGPBOLoop`; `GaussianKdeDensity`/
  `CategoricalDensity`/`LogDensityRatio`/`RunTPELoop` (Tree-structured Parzen Estimator,
  handling mixed continuous/categorical spaces GP-BO's own kernel cannot).
- **Bandit-based early stopping**: `ResumableTrial`/`TrialFactory` (the caller-owned
  incremental-training abstraction every algorithm below builds on),
  `RunSuccessiveHalving` (train-prune-repeat rung schedule), `RunHyperband` (multiple
  Successive Halving brackets trading off aggressiveness vs. budget), `RunASHA`
  (asynchronous variant: promotes a candidate the instant it qualifies, never waiting on the
  rest of its rung).
- Every algorithm here is also consumed directly by
  [Evolutionary Computation](../evolutionary-computation/index.md)'s own CMA-ES (via
  `DecodeGenotype` over this page's `SearchSpace`) and Population Based Training (via a
  `PBTResumableTrial` extending this page's own `ResumableTrial`).

Full API reference: [Doxygen: Hyperparameter Optimization](../api/group__hyperparameter__optimization.html)

## How to implement

### Random and grid search over a typed search space

```cpp
#include <random>

#include "pulsatrix/hpo_sampling.hpp"
#include "pulsatrix/search_space.hpp"

using namespace pulsatrix;

SearchSpace space;
space.AddLogUniform("learning_rate", 1e-4, 1.0);
space.AddInteger("hidden_size", 4, 64);

std::mt19937 rng(42);
Configuration config = RandomSample(space, rng);
float lr = static_cast<float>(std::get<double>(config.at("learning_rate")));
```

**What's happening:** `RandomSample` draws each parameter according to its own kind —
`LogUniform` samples uniformly in log-space (so `0.001` and `0.01` are equally likely orders
of magnitude, not equally likely absolute distances), `Integer` samples uniformly over its
inclusive bounds. `GridSample` is the deterministic sibling for exhaustive sweeps.

### Bandit-based early stopping: Successive Halving

```cpp
#include "pulsatrix/successive_halving.hpp"

using namespace pulsatrix;

class MyTrial : public ResumableTrial {
public:
    // ...owns whatever network/optimizer/dataset this configuration needs...
    double TrainForEpochs(int num_epochs) override {
        // train `num_epochs` more epochs, continuing from wherever training left off
        return /* current validation metric, maximization convention */ 0.0;
    }
};

TrialFactory make_trial = [](const Configuration& config) {
    return std::make_unique<MyTrial>(/* ...configured from config... */);
};

std::mt19937 rng(42);
SuccessiveHalvingResult result =
    RunSuccessiveHalving(space, make_trial, /*num_configs=*/16, /*initial_epoch_budget=*/1,
                          /*eta=*/2.0, rng);
```

**What's happening:** every candidate configuration trains for a small epoch budget, the
worst fraction (`1/eta`) is pruned, and the budget grows by `eta` each rung — until one
configuration survives. `ResumableTrial` is the caller-owned abstraction that makes "stop
training early" possible with zero changes to any training loop, `Module`, or optimizer: it's
an ordinary synchronous training loop the caller simply chooses not to call again for pruned
configurations.
