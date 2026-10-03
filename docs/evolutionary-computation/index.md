# Evolutionary Computation

Use this section when you want to optimize something by evolving a population of candidates
instead of following a gradient. It helps when the objective has no gradient, is noisy, or
involves structure such as a network's topology.

Pulsatrix provides a genetic-algorithm core (population, fitness, selection, crossover,
mutation) and several methods built on it: neuroevolution (NEAT, Evolution Strategies),
evolutionary hyperparameter search (CMA-ES), Population Based Training, and evolutionary GAN
training (E-GAN). Everything is C++, with no Python or DEAP dependency.

## Which method should I use?

| Goal | Use |
|---|---|
| Optimize a vector of numbers or bits against any fitness function | GA core: `RunEvolutionaryLoop` |
| Trade off several objectives at once | NSGA-II: `NSGA2Replacement` |
| Evolve a network's weights *and* its topology | NEAT: `RunNEATEvolution` |
| Train a fixed network's weights without backpropagation | Evolution Strategies: `RunEvolutionStrategies` |
| Tune continuous hyperparameters | CMA-ES: `CMAES` + `DecodeGenotype` |
| Tune hyperparameters while models train, copying weights from better runs | Population Based Training: `RunPBT` |
| Train a GAN with a population of generators | E-GAN: `RunEGANTraining` |

## What's inside

- **GA core**
    - `Individual` (genes plus a fitness value; higher fitness is better).
    - Selection: `TournamentSelect`, `RouletteSelect`, `RankSelect`.
    - Crossover: `OnePointCrossover`, `TwoPointCrossover`, `UniformCrossover`, `BlendCrossover`,
      `SimulatedBinaryCrossover`.
    - Mutation: `BitFlipMutation`, `GaussianMutation`, `PolynomialMutation`.
    - Survivor selection: `GenerationalReplacement`, `MuPlusLambdaReplacement`,
      `MuCommaLambdaReplacement`.
    - `RunEvolutionaryLoop`: the generation loop. Pass a `DataThreadPool` to evaluate fitness in
      parallel.
    - Multi-objective NSGA-II: `FastNonDominatedSort`, `CrowdingDistance`, `NSGA2Replacement`.
- **Neuroevolution**
    - NEAT: `NEATGenome` (a network encoded as node and connection genes, with innovation
      numbers that track when each gene appeared), `EvaluateNEATPhenotype` (runs a genome as a
      network), `CompatibilityDistance` / `SpeciatePopulation` / `ComputeAdjustedFitness`
      (groups similar genomes into species and shares fitness within each), `RunNEATEvolution`.
    - Evolution Strategies: `ESUpdateGivenPerturbations`, `ESStep`, `RunEvolutionStrategies`.
      Each random perturbation is also tried in the opposite direction (mirrored sampling).
      `FixedTopologyXORFitness` is a ready-made XOR fitness function for it.
- **Evolutionary hyperparameter search**: `DecodeGenotype` maps a vector of numbers in `[0, 1]`
  onto a `SearchSpace` (Continuous, LogUniform, Integer, Categorical). `CMAES` is a CMA-ES
  variant with a diagonal covariance. Both work with the `SearchSpace` and `Trial` types from
  [Hyperparameter Optimization](../hyperparameter-optimization/index.md).
- **Population Based Training**: `PBTResumableTrial` extends
  [Hyperparameter Optimization](../hyperparameter-optimization/index.md)'s `ResumableTrial` so
  weights and hyperparameters can be read and overwritten. `RunPBTGeneration` / `RunPBT`
  replace the worst trials with copies of the best (exploit), then perturb their
  hyperparameters (explore).
- **E-GAN**: `GeneratorPopulation` (trains a population of generators),
  `MutationObjective` / `MutationLoss` (`Minimax`, `Heuristic`, `LeastSquares`),
  `QualityFitness` / `DiversityFitness`, `RunEGANGeneration` / `RunEGANTraining`.

Full API reference: [Doxygen: Evolutionary Computation](../api/group__evolutionary.html)

## How to implement

### A genetic algorithm over real-valued vectors

```cpp
#include <random>
#include <vector>

#include "pulsatrix/crossover.hpp"
#include "pulsatrix/evolutionary_loop.hpp"
#include "pulsatrix/mutation.hpp"
#include "pulsatrix/selection.hpp"
#include "pulsatrix/survivor_selection.hpp"

using namespace pulsatrix;
using Genes = std::vector<double>;
using Ind = Individual<Genes, double>;

std::mt19937 rng(42);
std::normal_distribution<double> init(0.0, 2.0);

// 20 random 5-dimensional individuals.
std::vector<Ind> population(20);
for (Ind& ind : population) {
    for (int d = 0; d < 5; ++d) ind.genes.push_back(init(rng));
}

// Fitness is maximized: here, the closer to the origin the better.
auto fitness = [](const Genes& x) {
    double sum = 0.0;
    for (double v : x) sum -= v * v;
    return sum;
};

// Build one child: pick two parents by tournament, blend them, then mutate.
auto make_child = [&](const std::vector<Ind>& pop) {
    const Genes& a = pop[TournamentSelect(pop, /*tournament_size=*/3, rng)].genes;
    const Genes& b = pop[TournamentSelect(pop, /*tournament_size=*/3, rng)].genes;
    Genes child = BlendCrossover(a, b, /*alpha=*/0.5, rng).first;
    return GaussianMutation(child, /*sigma=*/0.1, /*mutation_probability=*/0.2, rng);
};

std::vector<Ind> final_pop = RunEvolutionaryLoop(
    population, /*num_generations=*/100, /*lambda_size=*/40, fitness, make_child,
    MuPlusLambdaReplacement<Genes, double>);   // keep the best 20 of parents + children
```

**What's happening:** each generation, `RunEvolutionaryLoop` calls `make_child` 40 times,
scores every child, and passes parents and children to the survivor selector. You choose the
operators by writing `make_child`. Swap in `RouletteSelect`, `SimulatedBinaryCrossover`, or
`PolynomialMutation` without changing the loop. To evaluate fitness in parallel, pass a
`DataThreadPool*` as the last argument; your fitness function must then be thread-safe.

### Neuroevolution: evolving a network from minimal topology

```cpp
#include <random>
#include <vector>

#include "pulsatrix/neat_evolution.hpp"
#include "pulsatrix/neat_genome.hpp"
#include "pulsatrix/neat_xor_fitness.hpp"

using namespace pulsatrix;

InnovationTracker tracker(/*next_node_id=*/4);   // ids 0-3: 2 inputs, 1 bias, 1 output
std::vector<NEATGenome> population;
for (int i = 0; i < 150; ++i) {
    population.emplace_back(/*num_inputs=*/2, /*num_outputs=*/1, /*has_bias=*/true, tracker);
}

std::mt19937 rng(42);
NEATEvolutionResult result =
    RunNEATEvolution(population, XORFitness, /*num_generations=*/200,
                     /*compatibility_threshold=*/3.0, /*c1=*/1.0, /*c2=*/1.0, /*c3=*/0.4,
                     /*weight_mutation_sigma=*/0.5, /*weight_mutation_probability=*/0.8,
                     /*add_connection_probability=*/0.05, /*add_node_probability=*/0.03,
                     tracker, rng);
// result.best_genome: the best genome found; run it with EvaluateNEATPhenotype
// result.best_fitness: its fitness (4.0 means XOR is solved exactly)
```

**What's happening:** every genome starts minimal: inputs wired straight to the output, with
no hidden nodes. Each generation, NEAT scores every genome and groups similar genomes into
species. Genomes then share fitness within their species, so a new structure gets time to
improve before it must compete with the whole population. Reproduction adds nodes and
connections by mutation. XOR cannot be solved without a hidden node, so solving it shows the
topology grew.

NEAT crossover is not implemented yet; reproduction is mutation-only.

### Evolutionary hyperparameter search: CMA-ES over a search space

```cpp
#include <algorithm>
#include <random>
#include <vector>

#include "pulsatrix/cma_es.hpp"
#include "pulsatrix/hpo_genotype.hpp"
#include "pulsatrix/search_space.hpp"

using namespace pulsatrix;

SearchSpace space;
space.AddLogUniform("learning_rate", 1e-4, 1.0);

CMAES cmaes(/*initial_mean=*/{0.5}, /*initial_sigma=*/0.2, /*lambda=*/10);  // 10 candidates per round
std::mt19937 rng(2026);

for (int generation = 0; generation < 20; ++generation) {
    std::vector<std::vector<double>> genotypes = cmaes.Ask(rng);  // samples around the mean
    std::vector<double> fitness(genotypes.size());
    for (size_t i = 0; i < genotypes.size(); ++i) {
        std::vector<double> genes = genotypes[i];
        for (double& g : genes) g = std::clamp(g, 0.0, 1.0);    // DecodeGenotype needs [0, 1]
        Configuration config = DecodeGenotype(space, genes);     // gene -> learning_rate
        double lr = std::get<double>(config.at("learning_rate"));
        fitness[i] = -(lr - 0.01) * (lr - 0.01);                 // higher is better; use -validation_loss
    }
    cmaes.Tell(genotypes, fitness);   // moves the mean toward the fitter candidates
}
```

**What's happening:** `Ask()` draws `lambda` candidates from a Gaussian around the current mean.
The samples are not bounded, so clamp each gene to `[0, 1]` before decoding;
`DecodeGenotype` throws otherwise. `Tell()` moves the mean toward the best candidates and adapts
the step size and per-dimension spread. Pass `Tell()` the original, unclamped genotypes.

`DecodeGenotype` maps each gene onto its parameter's declared bounds. For Continuous and
LogUniform parameters this matches GP-BO's `UnitCubeToConfiguration`. The result is an ordinary
`Configuration`, the same type every HPO algorithm and `Trial` uses.
