# Getting Started

This page takes you from a fresh clone to a working build, and covers the optional GPU, Python
and visualization builds.

## Prerequisites

- [CMake](https://cmake.org/download/) 3.20 or newer
- A C++17 compiler:
    - **Windows**: Visual Studio 2022 or newer (tested in CI)
    - **Linux**: GCC 11+ (tested in CI) or Clang 14+
    - **macOS**: Clang 14+ (Xcode 14+), which should work but isn't tested in CI
- Network access the first time you configure. CMake downloads GoogleTest, plus pybind11 or
  the GUI libraries if you turn those options on.

## Build and test

```bash
git clone https://github.com/Joshuaweg/pulsatrix.git
cd pulsatrix
```

**Linux / macOS**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Makefiles and Ninja build one configuration per directory. For a Debug build, use a second
directory, e.g. `cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug`.

**Windows**

```powershell
cmake -S . -B build -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

CMake picks the installed Visual Studio version automatically, so you don't need `-G`.
Visual Studio builds hold every configuration in one directory, and you choose one with
`--config` and `-C`.

## Run your first example

```bash
./build/xor_demo                 # Linux / macOS
build\Release\xor_demo.exe       # Windows
```

`xor_demo` trains a two-layer network on XOR and prints the loss as it falls, then the
network's predictions. The core of it looks like this (full source:
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
Tensor pred = net.forward(input);
// net.train_step(input, target, optimizer, sink, step) runs forward, backward and one Adam update
```

Every demo is built the same way, and you can build a single one with `--target`:

```bash
cmake --build build --target xor_demo --config Release
```

Where to go next:

- [Examples](https://github.com/Joshuaweg/pulsatrix/tree/master/examples): the full list of demos.
- [Recipes](recipes/index.md): short programs that each show one feature, with a walkthrough.
- [LRP](interpretability/lrp.md): explain your model's predictions.

## Build options

Pass `-D<OPTION>=ON` or `OFF` when you configure.

| Option | Default | What it does |
|---|---|---|
| `PULSATRIX_BUILD_TESTS` | `ON` | Builds the GoogleTest suite. |
| `PULSATRIX_BUILD_EXAMPLES` | `ON` | Builds the demos in `examples/` and the recipes in `examples/recipes/`. |
| `PULSATRIX_ENABLE_CUDA` | `OFF` | Builds the CUDA backend. See [GPU backends](#gpu-backends). |
| `PULSATRIX_ENABLE_HIP` | `OFF` | Builds the HIP/ROCm backend for AMD GPUs. See [GPU backends](#gpu-backends). |
| `PULSATRIX_ENABLE_PYTHON` | `OFF` | Builds the `pulsatrix_py` Python module. See [Python bindings](#python-bindings). |
| `PULSATRIX_ENABLE_VIZ` | `OFF` | Builds the Dear ImGui + ImPlot visualization module and its GUI demos. See [Visualization](visualization/index.md). |

## GPU backends

### CUDA

Install the CUDA Toolkit that matches your driver, then configure with
`-DPULSATRIX_ENABLE_CUDA=ON`. By default the build targets the GPU in your machine
(`CMAKE_CUDA_ARCHITECTURES=native`). Set `CMAKE_CUDA_ARCHITECTURES` yourself to build for
other GPUs.

### HIP / ROCm

Configure with `-DPULSATRIX_ENABLE_HIP=ON`. You need an AMD GPU and a ROCm install at
`/opt/rocm`, or set `ROCM_PATH`. The default target is `gfx1151`; set
`CMAKE_HIP_ARCHITECTURES` for other GPUs.

We recommend building inside the pinned ROCm container. `scripts/rocm-build.sh` runs a
command in it with your GPU passed through, and builds the image on first use:

```bash
scripts/rocm-build.sh 'cmake -S . -B build-hip -DCMAKE_BUILD_TYPE=Release -DPULSATRIX_ENABLE_HIP=ON'
scripts/rocm-build.sh 'cmake --build build-hip -j"$(nproc)"'
scripts/rocm-build.sh './build-hip/tests/pulsatrix_tests'
```

The container pins ROCm 7.2.4 because ROCm 7.1.x crashes on gfx1151, and Ubuntu's packages are
7.1.x. The header of `docker/Dockerfile.rocm` has the details. The HIP tests run on real
hardware, not a mock.

## Python bindings

The `pulsatrix_py` module exposes `Tensor`, the core layers, LRP and the other explainers, and
`SystemMonitor`. Point `PYTHON_EXECUTABLE` at a Python install that has development headers:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPULSATRIX_ENABLE_PYTHON=ON \
      -DPYTHON_EXECUTABLE="$(which python3)"
cmake --build build --target pulsatrix_py -j
PYTHONPATH=build python3 -c "import pulsatrix_py; print(pulsatrix_py.compiled_devices())"
```

With Visual Studio the module is usually written to `build/Release/`, so put that directory on `PYTHONPATH` instead.

A short example:

```python
import pulsatrix_py as px

fc1, relu, fc2 = px.LinearModule(2, 4), px.ReluModule(), px.LinearModule(4, 2)
fc1.set_weight([0.6, -0.3, 0.4, -0.7, 0.2, 0.5, -0.6, 0.1])
fc2.set_weight([0.5, -0.4, 0.3, 0.6, -0.2, 0.1, 0.4, -0.5])

ctx = px.ExplainerContext([fc1, relu, fc2])
x = px.Tensor.from_values([1, 2], [1.0, 0.5])
attr = px.LRP().explain(ctx, x, [1])
print(attr.method, [attr.values.at([0, i]) for i in range(2)])
```

Tensors support the buffer protocol, so `numpy.asarray(attr.values)` works on CPU tensors.
The tests in `tests/python/` show the rest of the API. Run them with
`PYTHONPATH=build python3 -m pytest tests/python` (needs `numpy` and `pytest`).

## MNIST data

The MNIST demos, recipes and tests need the MNIST files, which aren't in the repository.
Download them once with:

```bash
pip install torchvision
python3 tools/fetch_mnist.py     # Windows: py tools/fetch_mnist.py
```

This writes the original IDX files to `data/MNIST/raw/`. After that, nothing in the C++ build
or tests needs Python. Run the MNIST programs from the repository root so they find
`data/MNIST/raw/`.

## Using pulsatrix in your own project

There's no `install()` step yet, so `find_package(pulsatrix)` doesn't work. Add the repository
to your CMake tree and link `pulsatrix_core`:

```cmake
set(PULSATRIX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(PULSATRIX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/pulsatrix)
target_link_libraries(my_app PRIVATE pulsatrix_core)
```

## Building the API reference

The [API reference](api/index.html) is generated by Doxygen. The published site includes it.
To build it locally, install [Doxygen](https://www.doxygen.nl/) and run:

```bash
cmake --build build --target docs   # output: build/html/index.html
```

If you preview this site with `mkdocs serve`, the API reference links stay empty until you copy
`build/html/` into `site/api/`.

## Troubleshooting

**Configure fails while downloading GoogleTest.**
Check your network access to GitHub. If you only want the library and examples, configure with
`-DPULSATRIX_BUILD_TESTS=OFF`.

**`PULSATRIX_ENABLE_PYTHON=ON` doesn't find your Python.**
Pass `-DPYTHON_EXECUTABLE=<path>`, not `-DPython3_EXECUTABLE=<path>`. You can also set the
`PULSATRIX_PYTHON_EXECUTABLE` environment variable.

**MNIST tests or demos fail or find no data.**
Run `tools/fetch_mnist.py` (see [MNIST data](#mnist-data)) and run the program from the
repository root. CI skips the MNIST tests for the same reason.

**CMake can't find the CUDA toolkit.**
Install the CUDA Toolkit for your driver version and make sure CMake's
`find_package(CUDAToolkit)` can see it. On Linux this usually means `nvcc` is on your `PATH`.

**CMake can't find the `hip` or `hipblas` packages.**
Set `ROCM_PATH` to your ROCm install, or use the [ROCm container](#hip-rocm).
