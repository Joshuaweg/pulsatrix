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
| Can a layer's activations be split into sparser, more interpretable directions? | `SparseAutoencoder` |
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
