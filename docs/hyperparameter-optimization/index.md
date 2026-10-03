# Hyperparameter Optimization

Use this section to choose settings such as learning rate or layer width automatically instead
of by hand. You describe the allowed values once in a `SearchSpace`. Then you pick a search
algorithm: grid or random search, Bayesian optimization (a model of the objective that proposes
promising settings), or early stopping that cuts weak runs short. Every algorithm uses the same
`SearchSpace`, `Configuration`, and `Trial` types, and none needs Python.

## Which algorithm should I use?

| Situation | Use |
|---|---|
| Cheap objective, or you want a baseline | Random search: `RandomSample`. Small space: `GridSample`. |
| Expensive objective, few continuous parameters | GP-BO: `RunGPBOLoop` (Continuous and LogUniform parameters only) |
| Expensive objective with Integer or Categorical parameters | TPE: `RunTPELoop` |
| Iterative training where weak runs show up early | Successive Halving, Hyperband, or ASHA |
| Tuning while models train, copying weights between runs | Population Based Training (see [Evolutionary Computation](../evolutionary-computation/index.md)) |

## What's inside

- **Core**: `SearchSpace` (Continuous, LogUniform, Integer, and Categorical parameters),
  `Configuration` (a map from parameter name to chosen value), `Trial` (a configuration plus
  the metrics recorded for it), `RandomSample`, `GridSample`.
- **Bayesian optimization**
    - GP-BO: `GaussianProcessRegressor` (squared-exponential kernel; predicts a mean and
      variance), the acquisition functions `ExpectedImprovement`, `ProbabilityOfImprovement`, and
      `UpperConfidenceBound` (they score how promising an untried point is), and `RunGPBOLoop`.
      GP-BO supports Continuous and LogUniform parameters only. It throws on Integer or
      Categorical; use TPE for those.
    - TPE (Tree-structured Parzen Estimator): `GaussianKdeDensity`, `CategoricalDensity`,
      `LogDensityRatio`, `RunTPELoop`. It handles mixed continuous and categorical spaces.
- **Early stopping (bandit-based)**
    - `ResumableTrial` / `TrialFactory`: you implement training that can resume for more epochs.
      Every algorithm below builds on it.
    - `RunSuccessiveHalving`: train many configurations briefly, keep the best fraction, give
      them more epochs, repeat.
    - `RunHyperband`: runs several Successive Halving brackets, from many configurations with
      small budgets to few with large budgets.
    - `RunASHA`: promotes a configuration to the next budget as soon as it ranks in the top
      `1/eta` of its rung, without waiting for the rest of the rung. It runs single-threaded;
      "asynchronous" describes the promotion rule.

[Evolutionary Computation](../evolutionary-computation/index.md) reuses these types. Its
CMA-ES decodes into a `SearchSpace` via `DecodeGenotype`. Its Population Based Training uses
`PBTResumableTrial`, which extends `ResumableTrial`.

Full API reference: [Doxygen: Hyperparameter Optimization](../api/group__hyperparameter__optimization.html)

## How to implement

All algorithms *maximize*. If you are minimizing a loss, return `-loss`.

### Random and grid search over a typed search space

```cpp
#include <random>
#include <vector>

#include "pulsatrix/hpo_sampling.hpp"
#include "pulsatrix/search_space.hpp"

using namespace pulsatrix;

SearchSpace space;
space.AddLogUniform("learning_rate", 1e-4, 1.0);
space.AddInteger("hidden_size", 4, 64);

std::mt19937 rng(42);
Configuration config = RandomSample(space, rng);
float lr = static_cast<float>(std::get<double>(config.at("learning_rate")));
int64_t hidden = std::get<int64_t>(config.at("hidden_size"));

std::vector<Configuration> grid = GridSample(space, /*points_per_continuous_dimension=*/5);
```

**What's happening:** `RandomSample` draws each parameter according to its kind. `LogUniform`
samples evenly in log space, so `0.001` and `0.01` are equally likely. `Integer` samples
evenly over its inclusive bounds. `GridSample` lists every combination instead.
`points_per_continuous_dimension` (at least 2) sets how many evenly spaced values each
Continuous or LogUniform parameter gets. Integer and Categorical parameters always list every
value. Grid size grows multiplicatively with each parameter, so keep the space small.

### Bayesian optimization: GP-BO and TPE

```cpp
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "pulsatrix/gp_bo.hpp"
#include "pulsatrix/search_space.hpp"
#include "pulsatrix/tpe.hpp"

using namespace pulsatrix;

SearchSpace space;
space.AddLogUniform("learning_rate", 1e-4, 1.0);
space.AddContinuous("momentum", 0.0, 0.99);

// Return a value to maximize. For a loss, return -loss.
auto objective = [](const Configuration& config) {
    double lr = std::get<double>(config.at("learning_rate"));
    double momentum = std::get<double>(config.at("momentum"));
    return -std::pow(std::log10(lr) + 2.0, 2) - std::pow(momentum - 0.9, 2);  // stand-in for -val_loss
};

std::mt19937 rng(42);
std::vector<Trial> gp_trials =
    RunGPBOLoop(space, objective, /*num_initial_random=*/5, /*num_iterations=*/20,
                AcquisitionKind::ExpectedImprovement, /*num_candidates=*/500, rng);

std::vector<Trial> tpe_trials =
    RunTPELoop(space, objective, /*num_initial_random=*/10, /*num_iterations=*/40,
               /*gamma=*/0.2, /*num_candidates=*/100, rng);

// Each Trial records its score under the tag "objective".
double best = *gp_trials[0].LatestMetric("objective");
for (const Trial& t : gp_trials) best = std::max(best, *t.LatestMetric("objective"));
```

**What's happening:** both loops start with a few random trials. After that, each iteration
fits a model to every result so far and scores `num_candidates` random candidates. It then
evaluates the most promising one.

- **GP-BO** fits a Gaussian process (a model that predicts a value and its uncertainty). It
  picks the candidate with the best acquisition score, balancing "predicted good" against
  "still uncertain".
- **TPE** splits past trials into the best `gamma` fraction and the rest. It picks the candidate
  most likely under the good group relative to the bad one. It needs at least 2 initial trials.

Both return every `Trial` in the order it was run.

### Early stopping: Successive Halving, Hyperband, ASHA

```cpp
#include <memory>
#include <random>

#include "pulsatrix/asha.hpp"
#include "pulsatrix/hyperband.hpp"
#include "pulsatrix/search_space.hpp"
#include "pulsatrix/successive_halving.hpp"

using namespace pulsatrix;

class MyTrial : public ResumableTrial {
public:
    explicit MyTrial(double learning_rate) : learning_rate_(learning_rate) {}

    // Train num_epochs more epochs, continuing where the last call stopped.
    // Return the current validation metric (higher is better).
    double TrainForEpochs(int num_epochs) override {
        epochs_ += num_epochs;
        return -1.0 / epochs_ - learning_rate_;  // replace with your training loop
    }

private:
    double learning_rate_;
    int epochs_ = 0;
};

SearchSpace space;
space.AddLogUniform("learning_rate", 1e-4, 1.0);

TrialFactory make_trial = [](const Configuration& config) {
    return std::make_unique<MyTrial>(std::get<double>(config.at("learning_rate")));
};

std::mt19937 rng(42);
SuccessiveHalvingResult result =
    RunSuccessiveHalving(space, make_trial, /*num_configs=*/16, /*initial_epoch_budget=*/1,
                         /*eta=*/2.0, rng);
// result.best_configuration, result.best_metric, result.total_epochs_trained

HyperbandResult hb = RunHyperband(space, make_trial, /*max_resource=*/27, /*eta=*/3.0, rng);

ASHAResult asha = RunASHA(space, make_trial, /*max_configs_started=*/32,
                          /*initial_epoch_budget=*/1, /*eta=*/3.0, /*num_rungs=*/4, rng);
```

**What's happening:** in Successive Halving, every configuration first trains for
`initial_epoch_budget` epochs. Only the best `1/eta` fraction survives each round (a *rung*),
and the survivors' budget is multiplied by `eta`. This repeats until one configuration is
left. Hyperband runs several such brackets with different starting sizes, so you need not guess
how aggressive to be. ASHA promotes candidates one at a time as soon as they qualify.

`ResumableTrial` keeps the training loop in your hands. The algorithm only decides whether to
call `TrainForEpochs()` again, so pruning a configuration needs no change to your `Module`,
optimizer, or training code.
