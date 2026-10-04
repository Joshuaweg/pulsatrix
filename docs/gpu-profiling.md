# GPU Profiling (HIP)

`scripts/profile_hip.sh` runs a command under rocprofv3, groups its GPU kernels by op type and
writes a CSV of where the time went. It also reports the GPU's **busy time** over the run's wall
span. When that is low, the GPU is waiting on launches and synchronization, not computing.
Measure before optimizing: every HIP performance item on the [roadmap](roadmap/index.md#hip-training-efficiency-on-amd-gpus)
is judged against these numbers.

## Running it

rocprofv3 is in the ROCm container, so run the script there:

```bash
# build the fixed training workloads once
scripts/rocm-build.sh 'cmake -S . -B build-hip-release -DCMAKE_BUILD_TYPE=Release -DPULSATRIX_ENABLE_HIP=ON \
    && cmake --build build-hip-release -j"$(nproc)" --target hip_profile_workloads'

# profile one: <ops.csv> -- <command>
scripts/rocm-build.sh 'scripts/profile_hip.sh profile/cnn.csv -- ./build-hip-release/hip_profile_workloads cnn 20'
```

Any command works in place of `hip_profile_workloads`, such as your own training program. The
CSV has one row per op: `op, calls, total_us, mean_us, percent`. Op names come from the kernel
name (`batch_norm_forward_kernel` becomes `batch_norm_forward`). All of hipBLAS's GEMM kernels
are grouped as `gemm (hipBLAS)`, and HIP runtime helpers such as copies appear as `runtime: ...`.

`hip_profile_workloads` has three fixed workloads, each a full training step (forward, loss,
backward, AdamW):

- **`mlp`:** `Linear(256→512) → ReLU → Linear(512→512) → ReLU → Linear(512→10)`, batch 64.
- **`cnn`:** `Conv2D(3→16) → BatchNorm → ReLU → Conv2D(16→32, stride 2) → BatchNorm → ReLU →
  Flatten → Linear`, on 32 images of 3×32×32.
- **`tagger`:** the transformer tagger from the
  [full fine-tuning recipe](recipes/deep-learning/tagger_finetune.md).
- **`reduce`:** `DeviceBackend::dot` and `sum` over 16M floats, a microbenchmark for HIP-5.

## Baseline (gfx1151, ROCm 7.2.4, 20 steps)

Measured 2026-10-04 on the Radeon 8060S (Strix Halo), Release build, before any of the HIP
performance work.

| Workload | Step time | GPU busy | Biggest costs (share of kernel time) |
|---|---|---|---|
| `mlp` | about 1.0 ms | 12.9% | hipBLAS GEMM 56%, `fill` 11%, `column_sums` 8%, copies 7% |
| `cnn` | about 27 ms | 78.3% | `batch_norm_forward` 34%, `batch_norm_backward` 31%, hipBLAS GEMM 18%, `batch_norm_update_running` 14% |
| `tagger` | 7.8 ms | 11.6% | hipBLAS GEMM 38%, copies 15%, `fill` 14%, `column_sums` 12% |

What it shows:

- **The small models are bound by launch and sync overhead.** The `mlp` and `tagger` GPUs are busy
  only 12–13% of the time. The `tagger` launches 14,457 kernels in 20 steps, and every op waits
  for the stream to finish (HIP-4).
- **BatchNorm dominates the CNN.** Its three kernels take 79% of the kernel time, because each runs
  one thread per channel: 16 or 32 threads for 16,384 or 4,096 elements per channel (HIP-2).
- **Fills and copies are a large share.** Many of them are temporaries a caching allocator and
  fused kernels would avoid (HIP-3, HIP-6).

## Results

### HIP-5: `dot` and `sum` across many blocks

They used to run on a single 256-thread block. Now a first pass of up to 1024 blocks writes
per-block partial sums, and one block adds those in a fixed order: still deterministic, no
atomics. On the `reduce` workload (16M floats):

| Op | Before | After | Effective bandwidth after |
|---|---|---|---|
| `dot` (reads 128 MB) | 17.7 ms | 0.61 ms (29×) | 211 GB/s |
| `sum` (reads 64 MB) | 17.2 ms | 0.30 ms (58×) | 216 GB/s |

Both now run at the measured memory bandwidth of the gfx1151 (about 212 GB/s), so there is
nothing left to gain on these two ops.

### HIP-2: BatchNorm as parallel per-channel reductions

BatchNorm ran one thread per channel over every element of that channel. Now each per-channel
sum is a deterministic two-stage reduction: several blocks per channel produce partials, which one
thread per channel then adds in order. Normalization and the input gradient run one thread per
element. On the `cnn` workload:

| | Before | After |
|---|---|---|
| Step time | about 27 ms | about 10 ms (2.7×) |
| BatchNorm kernels (20 steps) | 342 ms, 79% of kernel time | about 3 ms |
| GPU busy | 78.3% | 45.8% |

GEMM is now 84% of the CNN's kernel time. The drop in GPU busy time means launch and sync
overhead is now the larger cost, which is HIP-4's target.

Results now differ from the CPU in the last bits. The GPU adds each channel's values by a tree,
while the CPU adds them one at a time. Both are deterministic. Softmax, layer and RMS norms, and
column sums keep their kernels: at current shapes each call takes about 5 µs, which is launch
overhead rather than missing parallelism (the roadmap's falsifier for HIP-2).
