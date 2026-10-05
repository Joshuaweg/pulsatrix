# Benchmarks

`pulsatrix_bench` measures three things on every backend a build has:

- **Training step time.** A full forward, loss, backward and AdamW step on three fixed models.
- **Explanation time.** LRP and Integrated Gradients on one input.
- **Explanation correctness.** LRP's conservation error and Integrated Gradients' completeness
  error, so performance work can't quietly break explanations.

It also compares two builds and reports regressions. The [GPU profiler](gpu-profiling.md)
says *where* GPU time goes; the benchmark suite says whether a change made things faster or
slower, and whether explanations are still right.

## Running it

`pulsatrix_bench` is built and installed with the library. A HIP or CUDA build also measures
that backend.

```bash
cmake --build build --target pulsatrix_bench
./build/pulsatrix_bench run                       # every device in this build
./build/pulsatrix_bench run --device hip --check  # one device, plus the correctness gate
./build/pulsatrix_bench run --quick -o report.json
```

| Option | Meaning | Default |
|---|---|---|
| `--device cpu\|hip\|cuda\|all` | Which backends to measure | `all` |
| `--filter TEXT` | Only benchmarks whose name contains `TEXT` | all |
| `--repeats N`, `--warmup N` | Timed and untimed runs per benchmark | 10, 2 (3, 1 with `--quick`) |
| `--check` | Exit 1 if an LRP conservation error is above 10⁻³ | off |
| `--label TEXT` | A name for the run, stored in the report | empty |
| `-o FILE` | Write a `pulsatrix.benchmark.v1` JSON report | none |

`--check` uses the relative conservation error that the Conv2D LRP tests accept per layer. It
doesn't depend on timing, so CTest runs it on the CPU in CI (`PulsatrixBench.CpuConservationCheck`).

### What it measures

| Benchmark | What | Unit |
|---|---|---|
| `train.mlp` | Linear(256→512) → ReLU → Linear(512→512) → ReLU → Linear(512→10), batch 64 | ms per step |
| `train.cnn` | Conv2D → BatchNorm → ReLU → Conv2D (stride 2) → BatchNorm → ReLU → Linear, 32 images of 3×32×32 | ms per step |
| `train.tagger` | The TRN-6 transformer tagger, the full training stack | ms per step |
| `explain.lrp_epsilon.mlp` | LRP, epsilon rule, on one input | ms |
| `explain.lrp_epsilon_plus.convnet` | LRP EpsilonPlus on the CNN without BatchNorm | ms |
| `explain.integrated_gradients.mlp` | Integrated Gradients, 32 steps | ms |
| `conservation.*` | \|Σ input relevance − explained logit\| / \|logit\| | relative error |
| `completeness.integrated_gradients.mlp` | \|Σ attributions − (f(x) − f(baseline))\| / \|f(x) − f(baseline)\| | relative error |

The models are fixed and deterministic (`tools/bench/workloads.hpp`). `hip_profile_workloads`
uses the same ones, so profiles and benchmarks describe the same work. Biases start at zero,
which makes LRP's conservation exact apart from rounding and the epsilon stabilizer. It also
makes the MLP positively homogeneous, so Integrated Gradients from a zero baseline is exact
apart from rounding.

### Baseline (gfx1151, ROCm 7.2.4, Release, 2026-10-04)

Measured before ROCm 10.0.0 became the default container; 10.0.0 is within 2% on every row (see
[GPU Profiling](gpu-profiling.md#rocm-1000-evaluation)).

| Benchmark | CPU | HIP |
|---|---|---|
| `train.mlp` | 74 ms | 0.38 ms |
| `train.cnn` | 96 ms | 5.3 ms |
| `train.tagger` | 0.38 ms | 2.5 ms |
| `explain.lrp_epsilon.mlp` | 0.77 ms | 0.29 ms |
| `explain.lrp_epsilon_plus.convnet` | 5.3 ms | 0.69 ms |
| `explain.integrated_gradients.mlp` | 36 ms | 7.2 ms |
| `conservation.lrp_epsilon.convnet` | 1.9 × 10⁻⁴ | 1.9 × 10⁻⁴ |

The tagger is too small for the GPU: launch overhead outweighs the work, which is
[HIP-6](roadmap/index.md#hip-training-efficiency-on-amd-gpus)'s target.

## Comparing two builds

```bash
scripts/bench_ab.sh build-master build-feature 6 -- --device hip
```

This runs the two builds' `pulsatrix_bench` alternately for 6 rounds and compares them. The
order is ABBA (old then new, then new then old), so drift in the machine's state affects both
builds equally. The reports are kept in `bench-ab/`. The script exits 1 if the new build
regressed. `pulsatrix_bench compare --baseline A.json... --candidate B.json...` compares any
reports directly.

A **time** regresses when both of these hold:

1. The candidate's median, over its rounds, is more than 10% slower (`--time-tolerance`).
2. With 3 or more rounds per side, the rounds separate. A one-sided Mann-Whitney U test must
   give p < 0.05 (`--significance`) that the candidate's rounds are slower.

A **metric** regresses when it rises by more than 10⁻⁵ (`--metric-tolerance`), or when it
becomes NaN or infinite.

!!! note "Why the significance test"
    Separate processes of the same build differ by up to 20% on sub-millisecond CPU work on this
    machine, even pinned to one core. Each whole run lands at one of two speeds. Comparing the
    build with itself flagged `explain.lrp_epsilon.mlp` 14 to 20% slower by median, but its
    rounds interleave (p = 0.35), so it is not reported. A real slowdown makes every round
    slower. Restoring HIP-4's per-op synchronization (`PULSATRIX_HIP_SYNC_DEBUG=1`) doubled
    `train.mlp`, and 4 rounds per side flagged it at p = 0.014.

    With fewer than 3 rounds per side there's no test, and the tolerance alone decides. Use
    more rounds when you can.
