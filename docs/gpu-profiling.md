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
