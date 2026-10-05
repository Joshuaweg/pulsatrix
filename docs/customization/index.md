# Customization: Extending pulsatrix

You extend pulsatrix by subclassing. To add a layer, a metrics sink or a device backend, you
subclass `Module`, `MetricsSink` or `DeviceBackend` and override their virtual methods. There is
no runtime plugin or factory registry: nothing like `Register("my_layer", ...)` and no
string-keyed lookups. [Why no runtime registry](#why-no-runtime-registry) explains the choice.

## Adding a new layer

Subclass `Module` ([`module.hpp`](https://github.com/Joshuaweg/pulsatrix/blob/master/include/pulsatrix/module.hpp))
and implement four pure-virtual methods:

```cpp
#include "pulsatrix/module.hpp"

class MyActivation : public Module {
public:
    explicit MyActivation(DeviceBackend* backend) : backend_(backend) {}

    [[nodiscard]] Tensor backward(const Tensor& grad_output) override {
        // gradient w.r.t. this module's input, from state cached in forward_impl()
    }

    [[nodiscard]] OpType op_type() const override { return OpType::Activation; }

    // Lets Module::forward() check that the input is on this layer's device.
    [[nodiscard]] std::optional<DeviceType> compute_device() const override { return backend_->device(); }

    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out,
                                              const LRPRuleConfig& config) override {
        // LRP: split relevance_out among this module's inputs (see below)
    }

protected:
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override {
        // the forward computation; Module::forward() calls this after checking the input
    }

private:
    DeviceBackend* backend_;
};
```

`compute_device()` is optional, but without it `Module::forward()` can't check that the input
lives on the same device as the layer, and a CPU tensor reaching a GPU layer fails much later
and less clearly. A layer with parameters usually returns its weight's device instead.

### The LRP rule is required

`propagate_relevance()` is pure virtual, so a layer without an LRP rule doesn't compile. This
is deliberate: pulsatrix never falls back to a generic rule that may not fit the layer.

- Use a rule from the literature and cite it in the header comment.
- If the right rule for your layer isn't known yet, throw from `propagate_relevance()` with a
  message that says so. Don't return a plausible-looking approximation.

### Supporting more than the epsilon rule

`LRPRuleConfig` tells your layer which rule to apply (`config.rule`) and that rule's parameters.
By default a `Module` supports only `LRPRule::Epsilon`. If your layer implements other rules,
override `supports_lrp_rule()`:

```cpp
[[nodiscard]] bool supports_lrp_rule(LRPRule rule) const override {
    return rule == LRPRule::Epsilon || rule == LRPRule::Gamma;
}
```

`LRP::explain()` checks every layer before it starts propagating, and throws if one doesn't
support the rule it was asked for. A layer that only passes relevance through unchanged, like
`ReluModule` or `FlattenModule`, can return `true` for every rule. The
[LRP guide](../interpretability/lrp.md) describes the existing rules.

If your layer has its own variants that aren't LRP rules, add an enum constructor parameter
rather than a new `LRPRuleConfig` field. `ConjunctionModule::TNorm`
(`Product`/`Lukasiewicz`/`Godel`, see [Neuro-Symbolic Reasoning](../neuro-symbolic/index.md))
is an example.

### Parameters

If your layer has trainable parameters, override `named_parameters()` (the default returns none)
so optimizers can update them and checkpoints, freezing and parameter groups can find them by
name. Use PyTorch's names where your layer has a PyTorch counterpart (`weight`, `bias`), and in a
layer that holds other modules, add each child's parameters under its name with
`append_named_parameters`:

```cpp
std::vector<NamedParamRef> named_parameters() override {
    std::vector<NamedParamRef> params{{"scale", {&scale_, &scale_grad_}}};
    append_named_parameters(params, "proj", proj_);  // proj.weight, proj.bias
    return params;
}
```

`parameters()` returns the same tensors in the same order without the names. Layers written
before `named_parameters()` existed may still override `parameters()` instead; they keep
training, but report no names of their own. Inside a container they get positional names
(`2.0`, `2.1`, ...), which are stable only as long as the parameter order is.

### Buffers and eval mode

State that isn't trained but must be saved, such as BatchNorm's running statistics, is a
buffer. Override `named_buffers()` so checkpoints store it; a layer that holds other modules
adds theirs with `append_named_buffers(out, "child", child_)`. Without it, a checkpoint
silently drops that state. If your layer holds other modules, also override `set_training()`
to pass the mode on to each of them, as `SequentialModule` does, so `set_training(false)`
reaches every BatchNorm and Dropout inside.

## Adding a new metrics sink

Training loops report metrics such as the loss through a `MetricsSink`. To send them somewhere
new, implement its two methods:

```cpp
#include "pulsatrix/metrics_sink.hpp"

class MyMetricsSink : public MetricsSink {
public:
    void log_scalar(const std::string& tag, double value, int step) override { /* ... */ }
    void log_histogram(const std::string& tag, const Tensor& values, int step) override { /* ... */ }
};
```

Pass your sink to `train_step()` (for example `XorNetwork::train_step()` or `MnistConvNet`'s).
Two sinks ship with the library: `NoOpMetricsSink`, which discards everything, and
`ImPlotMetricsSink`, which buffers series for live plots (see
[Visualization](../visualization/index.md)). An `ImPlotMetricsSink`'s log can be saved as a
`pulsatrix.training_log.v1` document (`ToTrainingLogDocument`) and replayed into any sink
later (`ReplayTrainingLog`).

## Adding a new device backend

Implement `DeviceBackend`, the interface that `CPUBackend` and the CUDA and HIP backends
implement. `Tensor` and `ComputationGraph` depend only on this interface, never on a specific
backend.

Every backend must behave the same way:

- **Results.** The same results as the CPU up to floating-point rounding (GPU reductions add in
  a different order), and the same results from run to run: no atomics in reductions.
- **Errors.** Throw on failure. `allocate()` returns nullptr only for a 0-byte request.
- **Ordering.** Ops may run asynchronously, but a `copy()` to the host, `dot()` and `sum()` must
  return finished results, and `free()` must be safe on memory that queued work still uses.
  The HIP backend shows one way: a single in-order stream, waits only where the host reads, and
  `PULSATRIX_HIP_SYNC_DEBUG=1` to wait after every op while hunting a fault
  ([GPU Profiling](../gpu-profiling.md#hip-4-no-per-op-synchronization)).
- **Memory.** `CachingAllocator` (`caching_allocator.hpp`) is device-independent: hand it your
  raw allocate, free and synchronize functions to reuse freed blocks
  ([GPU Profiling](../gpu-profiling.md#hip-3-a-caching-allocator)).
- **Determinism.** When `deterministic()` is on, use only paths whose results don't vary from
  run to run; the HIP and CUDA backends turn off BLAS atomics, for example.
- **Selection.** `top_k_rows()` ranks NaN above every number and keeps the lower index first on
  ties, so every backend picks the same entries in the same order.

This section is about writing a new backend. To build the existing CUDA or HIP backends, see
[Getting Started](../getting-started.md#gpu-backends).

## Why no runtime registry

A string-keyed registry would let you add layers without recompiling. The cost is that the
compiler could no longer check that every `Module` has an LRP rule. That check is the core
promise of the library, so every extension point uses ordinary subclassing and virtual
dispatch instead.
