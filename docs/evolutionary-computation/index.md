# Evolutionary Computation

A from-scratch, DEAP-free C++ genetic-algorithm core built on the
[Deep Learning Modules and Layers](../deep-learning/index.md) tensor core — population,
fitness, selection, crossover, and mutation, plus a representative spread of evolutionary
techniques built on top of it: neuroevolution (NEAT and Evolution Strategies), evolutionary
hyperparameter optimization, Population Based Training, and evolutionary generative-model
training (E-GAN). DEAP and Wang et al./Stanley & Miikkulainen's own papers were used as
research references only — every algorithm here is a from-scratch reimplementation, never a
runtime Python dependency.

## What's inside

- **GA core**: `Individual`, `TournamentSelect`/`RouletteSelect`/`RankSelect`, crossover
  (`OnePointCrossover`/`TwoPointCrossover`/`UniformCrossover`/`BlendCrossover`/
  `SimulatedBinaryCrossover`), mutation (`BitFlipMutation`/`GaussianMutation`/
  `PolynomialMutation`), survivor selection (`GenerationalReplacement`/
  `MuPlusLambdaReplacement`/`MuCommaLambdaReplacement`), `RunEvolutionaryLoop`
  (thread-pool-parallel fitness evaluation), and multi-objective NSGA-II
  (`FastNonDominatedSort`/`CrowdingDistance`/`NSGA2Replacement`).
- **Neuroevolution**: `NEATGenome` (connection-gene genome, innovation-number tracking,
  structural mutation), `EvaluateNEATPhenotype` (irregular-topology forward pass),
  `CompatibilityDistance`/`SpeciatePopulation`/`ComputeAdjustedFitness` (speciation + fitness
  sharing), `RunNEATEvolution`; and Evolution Strategies
  (`ESUpdateGivenPerturbations`/`ESStep`/`RunEvolutionStrategies`, mirrored/antithetic
  sampling, zero RL dependency).
- **Evolutionary hyperparameter optimization**: `DecodeGenotype` (genotype-to-configuration
  decode over Continuous/LogUniform/Integer/Categorical parameters) and `CMAES`
  (sep-CMA-ES-style separable covariance adaptation), both interoperable with
  [Hyperparameter Optimization](../hyperparameter-optimization/index.md)'s own `SearchSpace`/
  `Trial` types.
- **Population Based Training**: `PBTResumableTrial` (an extension of
  [Hyperparameter Optimization](../hyperparameter-optimization/index.md)'s own
  `ResumableTrial` adding live weight/hyperparameter read-write), `RunPBTGeneration`/`RunPBT`
  (truncation-selection exploit + hyperparameter explore).
- **Evolving generative models (E-GAN)**: `GeneratorPopulation` (population-of-generators
  training harness), `MutationObjective`/`MutationLoss` (Minimax/Heuristic/LeastSquares),
  `QualityFitness`/`DiversityFitness`, `RunEGANGeneration`/`RunEGANTraining`.

Full API reference: [Doxygen: Evolutionary Computation](../api/group__evolutionary.html)

## How to implement

### Neuroevolution: evolving a network from minimal topology

```cpp
#include <random>

#include "pulsatrix/neat_evolution.hpp"
#include "pulsatrix/neat_genome.hpp"
#include "pulsatrix/neat_xor_fitness.hpp"

using namespace pulsatrix;

InnovationTracker tracker(/*next_node_id=*/4);
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
// result.best_genome: the winning genome, decodable via EvaluateNEATPhenotype
// result.best_fitness: its own fitness (4.0 == exact XOR solve, per XORFitness's own scale)
```

**What's happening:** every genome starts minimal (fully connected, no hidden nodes); each
generation evaluates fitness, speciates the population (protecting structurally novel
genomes from immediate out-competition via fitness sharing), and reproduces via mutation
only — no crossover between genomes is built (a logged scope reduction; see the genome
header's own notes). Solving XOR (not linearly separable) from a hidden-node-free start is
NEAT's own classic validation task, confirming real structural growth occurred.

### Evolutionary hyperparameter optimization: CMA-ES over a real search space

```cpp
#include <random>

#include "pulsatrix/cma_es.hpp"
#include "pulsatrix/hpo_genotype.hpp"
#include "pulsatrix/search_space.hpp"

using namespace pulsatrix;

SearchSpace space;
space.AddLogUniform("learning_rate", 1e-4, 1.0);

CMAES cmaes(/*initial_mean=*/{0.5}, /*initial_sigma=*/0.2, /*population_size=*/10);
std::mt19937 rng(2026);

auto genotypes = cmaes.Ask(rng);                 // unit-hypercube candidate vectors
std::vector<double> fitness(genotypes.size());
for (size_t i = 0; i < genotypes.size(); ++i) {
    Configuration config = DecodeGenotype(space, genotypes[i]);  // -> real learning_rate
    fitness[i] = /* -validation_loss(config) */ 0.0;
}
cmaes.Tell(genotypes, fitness);                  // adapts mean/covariance for the next Ask()
```

**What's happening:** CMA-ES searches its own real-valued unit-hypercube genotype space
directly; `DecodeGenotype` (shared with the GA core's own genotype encoding) is the exact
same decoder [Hyperparameter Optimization](../hyperparameter-optimization/index.md)'s other
algorithms consume `SearchSpace` through, so a `Configuration` produced this way is
interoperable everywhere else in the library that expects one.
