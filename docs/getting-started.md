# Getting Started

## Clone

```bash
git clone https://github.com/Joshuaweg/pulsatrix.git
cd pulsatrix
```

## A minimal example

Train a tiny network on XOR and print predictions (the full, runnable version is
[`examples/xor_demo.cpp`](https://github.com/Joshuaweg/pulsatrix/blob/master/examples/xor_demo.cpp)):

```cpp
#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/xor_training_example.hpp"

using namespace pulsatrix;

CPUBackend backend;
XorNetwork net(&backend);              // Linear(2,4) -> ReLU -> Linear(4,1)
AdamOptimizer optimizer(0.01f, &backend);

Tensor input(Shape({1, 2}), &backend, {1.0f, 0.0f});
Tensor pred = net.forward(input);      // forward pass
// net.train_step(input, target, optimizer, sink, step) runs forward + backward + update
```

Build it and run it yourself:

```bash
cmake --build build --target xor_demo --config Release
build/Release/xor_demo.exe
```

See [`examples/README.md`](https://github.com/Joshuaweg/pulsatrix/blob/master/examples/README.md)
for the full list of 11 runnable demos (layers, sequence models, transformers, explainers, and
five RL algorithms).

## Prerequisites

- [CMake](https://cmake.org/download/) 3.20 or newer
- A C++17 compiler:
    - **Windows**: Visual Studio 2022+ (MSVC 19.4x) — CI-verified
    - **Linux**: GCC 11+ or Clang 14+ — CI-verified (GCC)
    - **macOS**: Clang 14+ (Xcode 14+) — community-untested, not yet CI-verified
- Network access at configure time (GoogleTest is fetched automatically via CMake `FetchContent`)

## Configure, build, test

**Windows:**

```bash
cmake -S . -B build -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

!!! note
    No `-G` is needed — CMake auto-detects whichever Visual Studio version is installed.

**Linux / macOS:**

Unix Makefiles/Ninja are single-config generators, so Debug and Release live in separate
build directories rather than one `-C <cfg>` selecting between them:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Build options

All default to the values shown; pass `-D<OPTION>=ON/OFF` at configure time to change them.

| Option | Default | Notes |
|---|---|---|
| `PULSATRIX_BUILD_TESTS` | `ON` | Builds the GoogleTest suite (fetched automatically). |
| `PULSATRIX_BUILD_EXAMPLES` | `ON` | Builds the demo executables in `examples/`. |
| `PULSATRIX_ENABLE_CUDA` | `OFF` | Builds the CUDA `DeviceBackend`. Requires the CUDA Toolkit. Defaults to `CMAKE_CUDA_ARCHITECTURES=native`. |
| `PULSATRIX_ENABLE_HIP` | `OFF` | Builds the HIP/ROCm `DeviceBackend`. Requires a ROCm toolchain. Defaults `CMAKE_HIP_ARCHITECTURES` to `gfx1151`. |
| `PULSATRIX_ENABLE_PYTHON` | `OFF` | Builds the pybind11 Python bindings. Requires `-DPYTHON_EXECUTABLE=<path>` (or `PULSATRIX_PYTHON_EXECUTABLE`) pointing at a Python install with dev headers. |

## Troubleshooting

- **Configure fails trying to fetch GoogleTest** — `PULSATRIX_BUILD_TESTS` is `ON` by default
  and clones GoogleTest from GitHub via `FetchContent` at configure time. Check network
  access, or configure with `-DPULSATRIX_BUILD_TESTS=OFF`.
- **`PULSATRIX_ENABLE_PYTHON=ON` doesn't pick up your Python install** — pass
  `-DPYTHON_EXECUTABLE=<path>`, not `-DPython3_EXECUTABLE=<path>`.
- **`mnist_*` tests fail or find no data** — run `py -3.11 tools/fetch_mnist.py` once
  (requires `torchvision`) to populate `data/MNIST/raw/`, then rebuild.
- **CUDA build can't find the toolkit** — install the CUDA Toolkit matching your driver,
  discoverable via `find_package(CUDAToolkit)`.
- **HIP build can't find `hip`/`hipblas`** — set `ROCM_PATH` (or install at the conventional
  `/opt/rocm`).

For the HIP/ROCm pinned-container workflow, see the
[README's HIP/ROCm section](https://github.com/Joshuaweg/pulsatrix#hiprocm-backend-pulsatrix_enable_hipon).
