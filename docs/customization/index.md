# Customization: Extending pulsatrix

pulsatrix has **no runtime plugin or factory registry** — there's no `Register("my_layer", ...)`
call or string-keyed lookup table anywhere in the codebase. Every extension point is
**compile-time polymorphism**: you subclass an abstract base class (`Module`, `MetricsSink`,
`DeviceBackend`) or select an already-defined variant through an enum (`ConjunctionModule::TNorm`,
`LRPRuleConfig`). This page collects those seams in one place — reading `module.hpp` alone
tells you *how* to subclass it, but not that this is a deliberate, project-wide convention
rather than something specific to layers.

## Adding a new layer

Subclass [`Module`](../deep-learning/index.md) and implement three pure-virtual methods:

```cpp
#include "pulsatrix/module.hpp"

class MyActivation : public Module {
public:
    explicit MyActivation(DeviceBackend* backend) : backend_(backend) {}

    [[nodiscard]] Tensor backward(const Tensor& grad_output) override {
        // gradient w.r.t. this module's input, from state cached in forward_impl()
    }

    [[nodiscard]] OpType op_type() const override { return OpType::Activation; }

    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out,
                                              const LRPRuleConfig& config) override {
        // LRP relevance propagation -- see below
    }

protected:
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override {
        // the actual forward computation; Module::forward() calls this after
        // precondition checks (empty-input rejection)
    }

private:
    DeviceBackend* backend_;
};
```

**`propagate_relevance()` is pure-virtual — not optional.** A module type without a defined LRP
rule is a compile error, not a runtime "no default rule" exception. This is deliberate: it's
the specific failure mode (silently falling back to a generic/undefined rule) this project
exists to avoid, the way Captum/Zennit-style post-hoc tools can. If a new layer's correct LRP
rule genuinely isn't known yet, `propagate_relevance()` should `throw` explicitly (see
`RWKVModule`/`RetNetModule` in [Deep Learning Modules and Layers](../deep-learning/index.md)
for real examples of a documented, intentional "not yet derived" throw) rather than being
skipped or stubbed with a plausible-looking approximation.

If your layer needs a **selectable rule variant** rather than one fixed formula (see
`ConjunctionModule::TNorm`'s `Product`/`Lukasiewicz`/`Godel` choice in
[Neuro-Symbolic Reasoning](../neuro-symbolic/index.md) for a worked example), add an enum
constructor parameter rather than reaching for `LRPRuleConfig` — that struct selects *which
rule a module uses*, project-wide (currently just `epsilon`), not a per-module-type option
list. Its own doc comment states the convention explicitly: new fields (gamma, alpha-beta,
...) get added when a module first actually needs them, not speculatively ahead of time.

If your layer has trainable parameters, override `parameters()` (default: none) so an
optimizer can update it uniformly alongside every other module.

## Adding a new metrics sink

Implement [`MetricsSink`](../visualization/index.md)'s two methods:

```cpp
#include "pulsatrix/metrics_sink.hpp"

class MyMetricsSink : public MetricsSink {
public:
    void log_scalar(const std::string& tag, double value, int step) override { /* ... */ }
    void log_histogram(const std::string& tag, const Tensor& values, int step) override { /* ... */ }
};
```

Every training loop in this codebase already threads a `MetricsSink&` through `train_step()`,
so a new sink needs no changes anywhere else — `ImPlotMetricsSink` (buffers series for live
plotting, see [Visualization](../visualization/index.md)) and `NoOpMetricsSink` (the charter's
stated minimum viable implementation) are the two concrete examples shipped today.

## Adding a new device backend

Implement [`DeviceBackend`](../deep-learning/index.md), the same interface `CPUBackend` (and
the CUDA/HIP backends) implement — `Tensor`/`ComputationGraph` depend only on this abstract
interface, never on a concrete backend's types. Every implementation must honor identical
numeric semantics and identical error behavior (throw on failure, never return null from
`allocate()`) — a Liskov Substitution contract, not just a suggestion. See
[Getting Started](../getting-started.md) for the `PULSATRIX_ENABLE_CUDA`/`PULSATRIX_ENABLE_HIP`
build flags an already-implemented backend is gated behind; this section is about writing a
*new* one, not configuring an existing one.

## Why no runtime registry

`MetricsSink`'s own doc comment names this pattern explicitly: "same OCP/DIP pattern as
`DeviceBackend`/`ExplainerContext`" — Open/Closed and Dependency Inversion, achieved through
ordinary virtual dispatch. A string-keyed factory/plugin registry would let extension happen
without recompiling, at the cost of losing the compile-time guarantee that every `Module` has a
real `propagate_relevance()` — the exact tradeoff `Module`'s own doc comment calls out as the
"Captum/Zennit failure mode this project exists to avoid." Since that guarantee is the
project's core explainability promise, every extension point here follows the same rule:
compile-time polymorphism, never a runtime lookup table.
