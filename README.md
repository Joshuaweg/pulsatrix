# exai_dl_library

ExAI-first C++ deep learning library — explainability as a first-class property of the computation graph, not a post-hoc wrapper.

Governed by `cpp_engineering.aDNA` (agent persona **Bjarne**):
- Charter: `../cpp_engineering.aDNA/what/docs/charter.md`
- Project state: `../cpp_engineering.aDNA/what/projects/exai_dl_library/STATE.md`
- Campaign roadmap: `../cpp_engineering.aDNA/how/campaigns/campaign_exai_dl_library_phase*/`
- Context library: `../cpp_engineering.aDNA/what/context/`

## Build

```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Requires: CMake 3.20+, a C++17 compiler (MSVC 19.4x verified), network access for GoogleTest via `FetchContent`.

On Linux the generator is single-config, so Debug and Release are separate build directories
rather than `-C <cfg>` against one:

```
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug && cmake --build build-debug -j"$(nproc)"
ctest --test-dir build-debug --output-on-failure
```

Verified on GCC 15.2.0 / CMake 4.2.3 and GCC 13.3.0 / CMake 3.28.3.

### HIP/ROCm backend (`EXAI_ENABLE_HIP=ON`)

Needs an AMD GPU and a ROCm toolchain. Use the pinned container — see the header of
`docker/Dockerfile.rocm` for why the ROCm version is pinned and why installing ROCm from a
distro package manager is not an equivalent substitute:

```
scripts/rocm-build.sh 'cmake -S . -B build-hip -DCMAKE_BUILD_TYPE=Debug -DEXAI_ENABLE_HIP=ON'
scripts/rocm-build.sh 'cmake --build build-hip -j"$(nproc)"'
scripts/rocm-build.sh './build-hip/tests/exai_tests'
```

`CMAKE_HIP_ARCHITECTURES` defaults to `gfx1151` (this project's dev device); override it for
another target. `ROCM_PATH` may be set if ROCm is not at `/opt/rocm`. The HIP tests run
against real hardware — there is no mocked path, deliberately.
