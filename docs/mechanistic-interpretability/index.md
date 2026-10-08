# Mechanistic Interpretability

Use this section when you want to see *how* a model computes its answer, not just which
inputs mattered. [Interpretability](../interpretability/index.md) explains
input-output behavior. The tools here open the network up: they cache its internal
activations, test what those activations encode, and measure which layers the output depends
on.

The section also hosts GFlowNets, a training method that learns to *sample* outcomes in
proportion to their reward instead of maximizing it. That makes them a tool for exploring a
model's or environment's structure.

## Which tool should I use?

| Question | Tool |
|---|---|
| What did each layer output for this input? | `ExplainerContext::activation_snapshot()` → `ActivationSnapshot` |
| Is concept X linearly readable from layer L? | `LinearProbe` |
| Can a layer's activations be split into sparser, more interpretable directions? | `SparseAutoencoder`, `TopKSparseAutoencoder` |
| Which layers does the output depend on for this input? | `ExplainerContext::build_circuit_graph()` → `CircuitGraph` |
| What happens to the output if I overwrite one activation? | `ExplainerContext::forward_pass_with_patch()` |
| What would the model predict if layer L were the last layer? | `ExplainerContext::logit_lens()` |
| Where does an attention layer look? | `ExplainerContext::attention_weights()` |
| How do I sample outcomes in proportion to a reward? | GFlowNet types below |

## What's inside

- **Activation access** (all on `ExplainerContext`): `activation_snapshot()` returns an
  `ActivationSnapshot`, a self-contained copy of one forward pass's activations that stays
  valid after later passes. `forward_pass_with_patch()` (activation patching), `logit_lens()`,
  and `attention_weights()` cover the rest of the table above.
- **Probing**: `LinearProbe` trains a linear classifier to test whether a binary concept is
  linearly decodable from a layer's activations.
- **Decomposition**: `SparseAutoencoder` reconstructs activations through a wider hidden layer
  with an L1 penalty, so each example uses only a few hidden units. It is the first
  [featurizer](#featurizers). `CircuitGraph` scores
  every node by how much zeroing it changes the output.
- **Baselines**: compare a probe's accuracy or a sparse autoencoder's statistics against a
  randomly re-initialized copy of the model with `NullModelBaseline`
  ([Interpretability](../interpretability/index.md#the-null-model-baseline)). A result that also
  shows up on the random model says nothing about what the trained one learned. A feature's
  statistics and top examples can be saved as a `pulsatrix.feature_dashboard.v1` document.
- **GFlowNet**: `HyperGridEnv`, `GFlowNetForwardPolicy`, `sample_gflownet_trajectory`
  (returns a `GFlowNetTrajectory`), `TrajectoryBalanceLoss`, `DetailedBalanceLoss`, and
  `SubTBLoss`. For SubTB(λ), you pass each sub-trajectory pair's λ-weight to `forward()`.
  `LearnableScalar` is the single trainable `log Z` value Trajectory Balance needs. It is not a
  `Module`.

Full API reference: [Doxygen: Mechanistic Interpretability](../api/group__mech__interp.html)

## How to implement

### Probing for a linearly decodable concept

```cpp
#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/explainer_context.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/linear_probe.hpp"
#include "pulsatrix/relu_module.hpp"

using namespace pulsatrix;

CPUBackend backend;
// hidden -> relu -> head: your trained network
LinearModule hidden(8, 64, &backend);
ReluModule relu(&backend);
LinearModule head(64, 2, &backend);
ExplainerContext ctx({&hidden, &relu, &head});

// 1. Cache activations for a batch of N inputs. Node 0 is the input; node i+1 is module i's output.
ctx.forward_pass(inputs);                                         // inputs: shape (N, 8)
ActivationSnapshot snapshot = ctx.activation_snapshot();
const Tensor& activation_batch = snapshot.activation(/*relu output=*/2);  // shape (N, 64)

// 2. Train a probe on those activations. label_batch: shape (N, 1), values in {0, 1}.
LinearProbe probe(/*activation_dim=*/64, &backend);
AdamOptimizer optimizer(0.01f, &backend);
for (int step = 0; step < 200; ++step) {
    probe.train_step(activation_batch, label_batch, optimizer);
}
float acc = probe.accuracy(activation_batch, label_batch);
```

**What's happening:** a `LinearProbe` is a `LinearModule(activation_dim, 1)` trained with
`BCEWithLogitsLoss` on `(activation, concept label)` pairs. High accuracy means the concept is
linearly decodable from that layer. Chance-level accuracy means it is not, at least not
linearly. The probe accepts any `(N, activation_dim)` batch, so you can also test it on
synthetic data first as a sanity check.

Recipe: [Sparse autoencoder + linear probe](../recipes/mechanistic-interpretability/sparse_autoencoder_probe.md).

### Featurizers

A featurizer rewrites activations `x` as sparse codes `f = encode(x)` over learned directions
and reconstructs `x ≈ decode(f)`: feature i writes along its decoder direction. Sparse
autoencoders, and the TopK, JumpReLU and block-sparse variants planned in the FEAT epic, share
one interface (`featurizer.hpp`), so training and metrics work for all of them:

```cpp
#include "pulsatrix/featurizer.hpp"
#include "pulsatrix/sparse_autoencoder.hpp"

SparseAutoencoder sae(/*dim=*/320, /*hidden_dim=*/4096, /*l1_lambda=*/1e-3f, &backend);
AdamOptimizer opt(1e-3f, &backend);
FeatureActivityTracker activity(sae.num_features());
for (const Tensor& batch : batches) {
    FeaturizerLoss loss = TrainFeaturizer(sae, batch, opt, /*unit_norm_decoder=*/true, &activity);
}
double l0 = MeanL0(sae, held_out);                      // active features per input
std::vector<int64_t> dead = activity.dead(1'000'000);   // silent for the last million inputs
std::vector<int64_t> dense = activity.dense(0.1);       // firing on more than 10% of inputs
```

- **Unit-norm decoders.** After every step each decoder direction is rescaled to unit length.
  The scale moves into the encoder, which a ReLU passes through unchanged, so reconstructions
  don't move.

  Without it, an L1 penalty is largely paid by shrinking every code and growing the decoder.
  On the control data in `tests/sparse_autoencoder_test.cpp`, a penalty of 0.05 with free
  decoders cuts the mean activation to 0.19 of the unpenalized run's but L0 only to 0.58. With
  unit-norm decoders it cuts L0 to 0.21. Judge sparsity by L0, not the mean activation.
- **Dead and dense latents.** `FeatureActivityTracker` counts how often each feature fires and
  when it last did.
  - Dead features never fire, so they never learn.
  - Dense ones fire on most inputs and usually encode a bias, not a concept.

  Both waste capacity; report them with every featurizer.
- **The SAE is a Module.** It maps input to reconstruction, with `backward()`, LRP and
  parameters named `encoder.*` and `decoder.*`, so checkpoints and optimizers work as for any
  model.

#### TopK sparse autoencoders

`TopKSparseAutoencoder` (Gao et al., arXiv 2406.04093) keeps each input's k largest latents
and zeroes the rest, so L0 is k by construction and there is no L1 penalty:

```cpp
#include "pulsatrix/topk_sparse_autoencoder.hpp"

TopKSaeOptions options;
options.k = 32;                     // active latents per input
options.dead_after = 1'000'000;     // inputs without firing before a latent counts as dead
TopKSparseAutoencoder sae(/*dim=*/320, /*num_features=*/4096, &backend, options);
sae.initialize_bias(first_batch);   // b_dec starts at the data's mean
for (const Tensor& batch : batches) {
    FeaturizerLoss loss = TrainFeaturizer(sae, batch, opt);  // loss.sparsity is the AuxK term
}
std::vector<int64_t> dead = sae.dead_latents();
```

- **No shrinkage.** An L1 penalty pulls every code toward zero. On sums of 3 of 12 known
  directions (`tests/topk_sparse_autoencoder_test.cpp`), an L1 SAE's reconstructions have 0.88
  of the data's norm; TopK's have 1.00, and it recovers at least 11 of the 12 directions.
- **Dead latents and AuxK.** A latent outside every input's top k gets no gradient. The
  auxiliary loss has the k_aux largest dead latents (default: half the input dimension)
  reconstruct the main reconstruction's error, weighted by 1/32. It only reaches latents whose
  pre-activation is positive, since TopK's ReLU still applies. In the test, 30 of 64 latents are
  dead after training with it and 48 without.
- **Decoder directions** are unit vectors after every step, and the decoder's gradient loses
  its component along each one. Unlike `SparseAutoencoder`, the scale isn't moved into the
  encoder: that would change which latents win the top k.
- **Checked against PyTorch.** The loss, every gradient and two Adam steps match a PyTorch
  rendering of the method (`tools/golden/make_topk_sae_golden.py`) to 1e-5.
- **Resuming training.** Checkpoints hold the parameters. Save `inputs_since_fired()` too, and
  restore it with `set_inputs_since_fired()`, or every latent starts out alive again.

#### Measuring a featurizer

`featurizer_metrics.hpp` holds SAEBench's core measures (Karvonen et al., arXiv 2503.09532).
None of them says a featurizer is useful, and its authors find that they don't reliably predict
usefulness. Each needs a baseline: the same featurizer trained on a randomly initialized model's
activations (`NullModelBaseline`), and, for a concept, a linear probe.

```cpp
#include "pulsatrix/featurizer_metrics.hpp"

// Reconstruction on held-out activations.
ReconstructionMetrics r = EvaluateReconstruction(sae, held_out);
// r.explained_variance, r.cosine, r.norm_ratio (below 1: shrunk codes), r.l0,
// r.dead_fraction (never fired), r.dense_fraction (fired on more than 10% of inputs)

// Loss recovered: splice the reconstructions back in at a position (0 is the embeddings,
// i is layer i's output) and compare the model's loss with the clean and zero-ablated ones.
LossRecovered lr = MeasureLossRecovered(sae, /*position=*/4, [&](const HiddenStateHook& hook) {
    model.set_hidden_state_hook(hook);
    double loss = HeldOutLoss(model);
    model.set_hidden_state_hook({});
    return loss;
});
// lr.recovered = (ablated - spliced) / (ablated - clean)

// Feature absorption for one concept (labels 0/1 per input), with a linear-probe baseline.
AbsorptionResult a = FeatureAbsorption(sae, held_out, labels);
// a.probe_f1, a.main_features, a.main_f1, a.absorption_rate, a.absorbing_features
```

- **Splicing.** `CausalLM` and `EncoderLM` call a `HiddenStateHook` at every position during
  `forward()`, and continue with whatever it returns. `SpliceHook` and `AblationHook` build the
  two replacements. A `rows` mask limits them to the rows the featurizer was trained on, for
  example residues but not `<cls>` and `<eos>`.
- **Absorption** (Chanin et al., arXiv 2409.14507). A feature that stands for a concept stays
  silent where a more specific feature fires instead and carries the concept's direction: a
  "starts with S" feature that skips "short". The steps:
  1. A logistic probe finds the concept's direction.
  2. The main features are those that each raise F1 by at least 0.03.
  3. A held-out positive the probe finds counts as absorbed when no main feature fires on it,
     and other features aligned with the probe (cosine at least 0.025) carry at least 40% of
     its projection onto the probe.

  The test plants a parent concept with ten rare children and finds every absorbed input.

**On ESM-2 8M.** `pulsatrix_probe_esm --sae` reports all of these for an SAE on layer 4's
residues (2,000 Swiss-Prot proteins, 2,560 latents), beside the same SAE trained on a randomly
initialized ESM-2:

| | L1 SAE | TopK SAE (k = 16) | L1, random model | TopK, random model |
|---|---|---|---|---|
| L0 | 17.0 | 16.0 | 7.4 | 16.0 |
| Explained variance | 0.66 | 0.78 | 0.74 | 0.95 |
| Norm ratio `\|x̂\|/\|x\|` | 0.80 | 0.87 | 0.87 | 0.98 |
| Dead latents | 0% | 2.5% | 53% | 85% |
| Masked-LM loss: clean 2.37, zero-ablated 3.91 | 2.52 spliced | 2.44 spliced | | |
| Loss recovered | 0.90 | 0.96 | none | none |

- **TopK beats L1 at the same L0** on every measure, and its codes are shrunk less.
- **Read the baseline before the score.** The random model's SAEs explain *more* variance, with
  most latents dead: a random network's representations are easy to compress. Explained variance
  alone says nothing about whether the model learned anything.
- **Loss recovered needs a position that matters.** Zeroing layer 4 of the random model doesn't
  raise its loss (3.55 vs 3.57 clean), so there is nothing to recover, and `recovered` is NaN.

Absorption, where the probe finds the concept (F1 at least 0.5; below that the random model
"absorbs" as much, and the rate means nothing):

| Concept | Probe F1 | L1: absorbed (random model) | TopK: absorbed (random model) |
|---|---|---|---|
| Helix | 0.69 | 42% (10%) | 25% (15%) |
| Beta strand | 0.55 | 57% (1%) | 50% (8%) |
| Transmembrane | 0.79 | 34% (21%) | 30% (17%) |
| Signal peptide | 0.90 | 19% (26%) | 12% (37%) |
| Zinc finger | 0.62 | 30% (13%) | 12% (71%) |

- **The baseline decides here too.** Helix, strand and transmembrane residues missed by the
  concept's main features are carried by other aligned features well above the random model's
  rate: absorption, as Chanin et al. describe it. For signal peptides and zinc fingers the
  random model scores as high or higher, so those rates aren't evidence of anything.
- **TopK absorbs less than L1** for every concept here.


A featurizer is a discovery tool, not a detector: a low reconstruction error says nothing about
whether its directions mean anything. Compare features with probes and with a randomly
initialized model's.

### Building a circuit graph

```cpp
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/explainer_context.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/relu_module.hpp"

using namespace pulsatrix;

CPUBackend backend;
// hidden -> relu -> head: your trained network
LinearModule hidden(4, 16, &backend);
ReluModule relu(&backend);
LinearModule head(16, 3, &backend);
ExplainerContext ctx({&hidden, &relu, &head});

Tensor input(Shape({1, 4}), &backend, {0.2f, -1.0f, 0.5f, 0.9f});
CircuitGraph circuit = ctx.build_circuit_graph(input);
for (const CircuitNode& node : circuit.nodes()) {
    // node.ablation_effect: how far the output moves when this node is zeroed
}
```

**What's happening:** for each node, `build_circuit_graph()` reruns the forward pass with
that node's activation replaced by zeros. It records the L2 distance between the patched
output and the normal output as `ablation_effect`. A larger value means the output depends
more on that node for this input. The output node scores 0 by convention. Edges connect each
node to the next and carry the source node's score. `CircuitGraph` holds data only. To draw it,
use `CircuitGraphView` from [Visualization](../visualization/index.md), which also accepts the
saved form. To save it, `ToJson(ToCircuitGraphDocument(circuit))` writes a
`pulsatrix.circuit_graph.v1` document, and `ParseCircuitGraphDocument()` with `ToCircuitGraph()`
reads it back.

### Sampling a GFlowNet trajectory

```cpp
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/gflownet_forward_policy.hpp"
#include "pulsatrix/gflownet_trajectory.hpp"
#include "pulsatrix/hypergrid_env.hpp"
#include "pulsatrix/linear_module.hpp"

using namespace pulsatrix;

CPUBackend backend;
HyperGridEnv env(&backend, /*ndim=*/2, /*side_length=*/5);
LinearModule policy_net(2, 3, &backend);     // 2-dim state -> 3 actions (+x, +y, stop)
GFlowNetForwardPolicy policy(&policy_net, /*action_dim=*/3, &backend);

GFlowNetTrajectory traj = sample_gflownet_trajectory(env, policy);
// traj.states / traj.actions: every decision point and the action taken there
// traj.sum_log_pf, traj.sum_log_pb: the log-probability sums Trajectory Balance needs
// traj.terminal_reward: R(x) at the final state
```

**What's happening:** `sample_gflownet_trajectory` resets `env`. It then samples actions from
the policy, skipping invalid ones, and steps `env` until the episode ends. An episode ends on
an explicit `stop` action or at the environment's step cap. Along the way it sums the forward
and backward log-probabilities (`Σ log P_F`, `Σ log P_B`).

Feed the trajectory to `TrajectoryBalanceLoss`, `DetailedBalanceLoss`, or `SubTBLoss`. They
train the policy to sample each outcome `x` with probability proportional to `R(x)`, rather
than always picking the best one.

Recipe: [GFlowNet on HyperGrid](../recipes/mechanistic-interpretability/gflownet_hypergrid.md).

## Recipes

- [Sparse autoencoder + linear probe](../recipes/mechanistic-interpretability/sparse_autoencoder_probe.md)
- [GFlowNet on HyperGrid](../recipes/mechanistic-interpretability/gflownet_hypergrid.md)
