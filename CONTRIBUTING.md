# Contributing to Pulsatrix

## Building and testing

See the [README](README.md#build) for full build instructions. In short:

```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64   # Windows
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

## Development practice

This codebase is built test-first: every behavior change ships with a failing
test written before the implementation (red), then the minimal implementation
to pass it (green). PRs that add or change behavior should follow the same
pattern — please include tests, not just implementation.

A few conventions worth knowing before you dive in:
- Public headers live in `include/pulsatrix/`, implementations in `src/`.
- Every layer that participates in relevance propagation implements a real,
  cited LRP rule (see the README's "What's here" section for references) —
  no placeholder/no-op implementations.
- Numerical algorithms (RL agents, sequence models, etc.) use this codebase's
  deterministic LCG convention instead of `<random>` so runs are reproducible;
  follow the same pattern for new stochastic code.
- Extension always happens through compile-time polymorphism (subclassing
  `Module`/`MetricsSink`/`DeviceBackend`, or an enum-selected variant) — there
  is no runtime plugin/factory registry. See
  [Customization](https://joshuaweg.github.io/pulsatrix/customization/) for
  the full pattern and why, before adding a new layer, LRP rule variant,
  metrics sink, or device backend.

## Submitting a change

1. Fork the repo and create a branch for your change.
2. Write a failing test, then make it pass.
3. Run the full test suite (`ctest ...`) and confirm it's green.
4. Open a PR describing what changed and why.

## Reporting bugs

Open a GitHub issue with a minimal repro (a failing test is ideal) and the
platform/compiler you're building with.
