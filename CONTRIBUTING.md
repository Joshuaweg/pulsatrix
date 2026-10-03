# Contributing to Pulsatrix

Thanks for your interest. Bug reports, fixes, docs improvements and new features are all
welcome.

## Building and testing

The full instructions, including GPU and Python builds, are in
[Getting Started](https://joshuaweg.github.io/pulsatrix/getting-started/). In short:

```bash
# Linux / macOS
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure

# Windows
cmake -S . -B build -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

## How we work

**Tests come first.** Every behavior change starts with a test that fails, followed by the
smallest change that makes it pass. Please include tests with any PR that adds or changes
behavior.

Conventions to know before you start:

- **Layout.** Public headers live in `include/pulsatrix/`, implementations in `src/`, and tests
  in `tests/`.
- **Every layer needs an LRP rule.** `Module::propagate_relevance()` is a required method.
  Implement a rule from the literature and cite it in the header. If the right rule isn't known
  yet, throw with a clear message instead of guessing. See the
  [LRP guide](https://joshuaweg.github.io/pulsatrix/interpretability/lrp/) for the existing rules.
- **Deterministic randomness.** Stochastic code (RL agents, sequence models, and so on) uses a
  seeded linear congruential generator (LCG), as the existing RL code does, instead of
  `<random>`. That keeps runs reproducible. Follow the same pattern in new code.
- **Extend by subclassing.** New layers, metrics sinks and device backends subclass `Module`,
  `MetricsSink` or `DeviceBackend`. There is no runtime plugin registry.
  [Customization](https://joshuaweg.github.io/pulsatrix/customization/) explains the pattern
  and why.

## Submitting a change

1. Fork the repository and create a branch.
2. Write a failing test, then make it pass.
3. Run the full test suite and make sure it passes.
4. Open a pull request that explains what changed and why.

CI builds and tests every pull request on Windows and Linux.

## Reporting bugs

Open a [GitHub issue](https://github.com/Joshuaweg/pulsatrix/issues) with:

- a minimal way to reproduce the bug (a failing test is ideal),
- your OS, compiler and version, and
- the CMake options you used.

For security issues, follow [SECURITY.md](SECURITY.md) instead of opening a public issue.
