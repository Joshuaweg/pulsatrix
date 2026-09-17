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
