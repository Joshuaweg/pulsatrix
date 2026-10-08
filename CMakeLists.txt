cmake_minimum_required(VERSION 3.20)
project(pulsatrix VERSION 1.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(PULSATRIX_ENABLE_CUDA "Build the CUDA DeviceBackend" OFF)
option(PULSATRIX_ENABLE_HIP "Build the HIP/ROCm DeviceBackend" OFF)
option(PULSATRIX_ENABLE_PYTHON "Build Python bindings via pybind11" OFF)
option(PULSATRIX_ENABLE_VIZ "Build the Dear ImGui + ImPlot native visualization module" OFF)
option(PULSATRIX_BUILD_TESTS "Build the GoogleTest suite" ON)
option(PULSATRIX_BUILD_EXAMPLES "Build standalone example executables (examples/)" ON)
option(PULSATRIX_BUILD_FUZZERS "Build fuzz targets (tools/fuzz/); add sanitizer flags yourself" OFF)
option(PULSATRIX_FUZZ_STANDALONE "Fuzz targets get their own main() instead of libFuzzer's" OFF)
# On by default only when pulsatrix is the top-level project, so add_subdirectory and
# FetchContent users don't install pulsatrix alongside their own targets.
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(pulsatrix_install_default ON)
else()
    set(pulsatrix_install_default OFF)
endif()
option(PULSATRIX_INSTALL "Generate install rules and the find_package(pulsatrix) config" ${pulsatrix_install_default})

include(GNUInstallDirs)

if(PULSATRIX_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "native")  # build for the actual dev/CI GPU; override for multi-arch distribution
    endif()
endif()

if(PULSATRIX_ENABLE_HIP)
    # ROCm does not install its CMake package configs anywhere CMake searches by default
    # (they live under <rocm>/lib/cmake/{hip,hipblas,...}), and unlike CUDA there is no
    # FindHIPToolkit equivalent in CMake itself. This is the prefix-path setup AMD's own
    # documentation prescribes: honor an explicit -DROCM_PATH, then $ROCM_PATH, then the
    # conventional /opt/rocm. Nothing here is specific to a particular ROCm install.
    if(NOT DEFINED ROCM_PATH)
        if(DEFINED ENV{ROCM_PATH})
            set(ROCM_PATH "$ENV{ROCM_PATH}")
        else()
            set(ROCM_PATH "/opt/rocm")
        endif()
    endif()
    list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}" "${ROCM_PATH}/lib/cmake")

    enable_language(HIP)
    find_package(hip REQUIRED)
    find_package(hipblas REQUIRED)
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
        # This project's dev/CI AMD device. Unlike CUDA's "native", HIP has no portable
        # auto-detect here -- rocm_agent_enumerator exists but isn't a CMake facility --
        # so the target is named explicitly and overridden for multi-arch distribution.
        set(CMAKE_HIP_ARCHITECTURES "gfx1151")
    endif()
endif()

add_library(pulsatrix_core
    src/cpu_backend.cpp
    src/tensor.cpp
    src/sampler.cpp
    src/collate.cpp
    src/data_thread_pool.cpp
    src/data_loader.cpp
    src/csv_reader.cpp
    src/csv_dataset.cpp
    src/computation_graph.cpp
    src/autograd.cpp
    src/linear_module.cpp
    src/relu_module.cpp
    src/softmax_module.cpp
    src/rope_module.cpp
    src/multihead_attention_module.cpp
    src/swiglu_module.cpp
    src/transformer_block.cpp
    src/activation_module.cpp
    src/feed_forward_module.cpp
    src/encoder_block.cpp
    src/residual_module.cpp
    src/conjunction_module.cpp
    src/disjunction_module.cpp
    src/negation_module.cpp
    src/aggregator_module.cpp
    src/satisfaction_loss.cpp
    src/neuro_symbolic_toy_kb.cpp
    src/datalog_rule.cpp
    src/datalog_fact_database.cpp
    src/datalog_engine.cpp
    src/datalog_weighted_fact_database.cpp
    src/datalog_weighted_engine.cpp
    src/neuro_symbolic_datalog_bridge.cpp
    src/datalog_lrp.cpp
    src/conv2d_module.cpp
    src/rms_norm_module.cpp
    src/layer_norm_module.cpp
    src/group_norm_module.cpp
    src/batch_norm_module.cpp
    src/max_pool2d_module.cpp
    src/avg_pool2d_module.cpp
    src/embedding_module.cpp
    src/tied_lm_head_module.cpp
    src/ice.cpp
    src/sensitivity.cpp
    src/counterfactual.cpp
    src/global_sensitivity.cpp
    src/generation.cpp
    src/kv_cache.cpp
    src/hf_model.cpp
    src/causal_lm.cpp
    src/encoder_lm.cpp
    src/protein_concepts.cpp
    src/protein_contacts.cpp
    src/protein_explanations.cpp
    src/protein_sequences.cpp
    src/protein_structure.cpp
    src/protein_training.cpp
    src/variant_scoring.cpp
    src/fitness_metrics.cpp
    src/proteingym.cpp
    src/golden_logits.cpp
    src/attnlrp_parity.cpp
    src/relevance_graph.cpp
    src/dropout_module.cpp
    src/sequential_module.cpp
    src/top_k.cpp
    src/matrix_decompositions.cpp
    src/batch_norm_fold.cpp
    src/determinism.cpp
    src/safetensors.cpp
    src/checkpoint.cpp
    src/grad_clipping.cpp
    src/token_cross_entropy_loss.cpp
    src/tagger_finetune_example.cpp
    src/explanation_metrics.cpp
    src/featurizer.cpp
    src/featurizer_metrics.cpp
    src/topk_sparse_autoencoder.cpp
    src/jumprelu_sparse_autoencoder.cpp
    src/null_model_baseline.cpp
    src/caching_allocator.cpp
    src/rnn_module.cpp
    src/lstm_module.cpp
    src/gru_module.cpp
    src/mamba_module.cpp
    src/rwkv_module.cpp
    src/retnet_module.cpp
    src/reparameterize.cpp
    src/noise_schedule.cpp
    src/sinusoidal_timestep_embedding.cpp
    src/cartpole_env.cpp
    src/continuous_cartpole_env.cpp
    src/hypergrid_env.cpp
    src/learnable_scalar.cpp
    src/gflownet_forward_policy.cpp
    src/trajectory_balance_loss.cpp
    src/detailed_balance_loss.cpp
    src/subtb_loss.cpp
    src/gflownet_trajectory.cpp
    src/replay_buffer.cpp
    src/rollout_buffer.cpp
    src/dqn_loss.cpp
    src/dqn_target.cpp
    src/dqn_agent.cpp
    src/categorical_policy_agent.cpp
    src/policy_gradient_loss.cpp
    src/gae.cpp
    src/ppo_clipped_loss.cpp
    src/tanh_gaussian_policy.cpp
    src/polyak_update.cpp
    src/mse_loss.cpp
    src/calibration_loss.cpp
    src/cross_entropy_loss.cpp
    src/kl_divergence_loss.cpp
    src/bce_with_logits_loss.cpp
    src/mnist_loader.cpp
    src/mnist_classifier_example.cpp
    src/sgd_optimizer.cpp
    src/adam_optimizer.cpp
    src/xor_training_example.cpp
    src/image_decoder.cpp
    src/image_transforms.cpp
    src/image_folder_dataset.cpp
    src/tokenizer.cpp
    src/text_tokenizer.cpp
    src/tokenizer_components.cpp
    src/unicode.cpp
    src/unicode_regex.cpp
    src/byte_level_bpe.cpp
    src/sentencepiece_bpe.cpp
    src/tokenizer_json.cpp
    src/tokenizer_parity.cpp
    src/word_scores.cpp
    src/vocabulary.cpp
    src/text_dataset.cpp
    src/text_collate.cpp
    src/wav_reader.cpp
    src/audio_folder_dataset.cpp
    src/audio_transforms.cpp
    src/audio_collate.cpp
    src/video_frame_directory_dataset.cpp
    src/video_transforms.cpp
    src/dataset_validator.cpp
    src/lrp_conservation.cpp
    src/lrp_rules.cpp
    src/explainer_stability.cpp
    src/system_monitor.cpp
    src/system_monitor_linux.cpp
    src/system_monitor_host.cpp
    src/implot_metrics_sink.cpp
    src/viz/colormap.cpp
    src/viz/plot_data.cpp
    src/viz/document.cpp
    src/viz/svg.cpp
    src/viz/text_relevance.cpp
    src/viz/html.cpp
    src/viz/attribution_graph.cpp
    src/viz/protein_documents.cpp
    src/viz/protein_views.cpp
    src/viz/svg_whatif.cpp
    src/viz/document_whatif.cpp
    src/benchmark.cpp
    src/json.cpp
)
add_library(pulsatrix::core ALIAS pulsatrix_core)
set_target_properties(pulsatrix_core PROPERTIES EXPORT_NAME core)
target_include_directories(pulsatrix_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(pulsatrix_core PUBLIC cxx_std_17)

# First consumer: DataThreadPool/BoundedQueue (campaign_exai_dl_library_data_pipeline,
# Mission 2) -- pulsatrix's first std::thread-based code, needs the platform thread
# library linked explicitly (not implicit via any other dependency).
find_package(Threads REQUIRED)
target_link_libraries(pulsatrix_core PUBLIC Threads::Threads)

# SystemMonitor loads NVML at runtime (dlopen / LoadLibrary) -- never a link-time NVIDIA
# dependency. CMAKE_DL_LIBS is libdl on glibc < 2.34 and empty where dlopen is in libc;
# psapi (GetProcessMemoryInfo) is a Windows system library.
target_link_libraries(pulsatrix_core PRIVATE ${CMAKE_DL_LIBS})
if(WIN32)
    target_link_libraries(pulsatrix_core PRIVATE psapi)
endif()

# stb_image (public-domain, single-header image decoder) -- campaign_exai_dl_library_data_pipeline
# Phase 2 Decision Point 2: chosen over OpenCV (disproportionate build/dependency footprint
# for "decode a file into pixel bytes") and over hand-rolling (a real JPEG/PNG codec is a
# project-sized undertaking on its own). Fetched unconditionally, no PULSATRIX_ENABLE_*
# option gate -- unlike CUDA/HIP/Python, it needs no real hardware/SDK/interpreter, just a
# vendored header, so there is no real "opt out" case to support.
include(FetchContent)
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG f0569113c93ad095470c54bf34a17b36646bbbb
)
FetchContent_MakeAvailable(stb)
target_include_directories(pulsatrix_core PRIVATE ${stb_SOURCE_DIR})

if(PULSATRIX_ENABLE_CUDA)
    target_sources(pulsatrix_core PRIVATE src/cuda_backend.cu)
    # No cross-TU __device__/__global__ calls yet -- relocatable device code (and its
    # separate device-link step) isn't needed and was failing to link
    # (__cudaRegisterLinkedBinary unresolved) when pulsatrix_tests linked the static lib.
    target_link_libraries(pulsatrix_core PUBLIC CUDA::cudart CUDA::cublas)
endif()

if(PULSATRIX_ENABLE_HIP)
    target_sources(pulsatrix_core PRIVATE src/hip_backend.hip)
    # hip::host, not hip::device. hip::device carries --offload-arch as
    # INTERFACE_COMPILE_OPTIONS, which CMake applies to every language, so linking it makes
    # the plain C++ sources in this target (and in anything downstream) fail with
    # "c++: error: unrecognized command-line option '--offload-arch=gfx1151'". The HIP
    # language support enabled above already passes the architecture to the HIP compiler for
    # .hip sources, which is the only place it belongs. hip::host supplies the headers and
    # runtime without the device flags, and is PUBLIC because hip_backend.hpp is a public
    # header that includes <hip/hip_runtime.h>.
    target_link_libraries(pulsatrix_core PUBLIC hip::host roc::hipblas)
    # Deliberately NOT setting HIP_SEPARABLE_COMPILATION, for the same reason
    # CUDA_SEPARABLE_COMPILATION was removed above: there are no cross-TU
    # __device__/__global__ calls, so the separate device-link step buys nothing and broke
    # the link once a real device source existed.
endif()

if(PULSATRIX_ENABLE_PYTHON)
    # Multiple Python interpreters exist on a typical dev machine; the one on PATH is not
    # guaranteed to be the one with dev headers/import libs and the packages the pytest
    # suite needs. Pin explicitly rather than trusting the default PATH search -- see
    # campaign_exai_dl_library_phase5_bindings.md's activation Recon.
    # pybind11 v2.13.6 locates Python via the classic (deprecated but still what pybind11
    # itself uses) find_package(PythonInterp)/find_package(PythonLibs) modules, which read
    # PYTHON_EXECUTABLE -- NOT Python3_EXECUTABLE (the modern FindPython3 module's
    # variable). Passing only -DPython3_EXECUTABLE=... silently has no effect and
    # pybind11 falls back to whatever "python3" resolves to on PATH -- confirmed the hard
    # way at this mission's first configure attempt. Set both so this doesn't regress if
    # a future pybind11 upgrade switches modules.
    if(NOT DEFINED PYTHON_EXECUTABLE)
        set(PYTHON_EXECUTABLE "$ENV{PULSATRIX_PYTHON_EXECUTABLE}")
    endif()
    if(NOT DEFINED Python3_EXECUTABLE)
        set(Python3_EXECUTABLE "$ENV{PULSATRIX_PYTHON_EXECUTABLE}")
    endif()
    if(NOT PYTHON_EXECUTABLE)
        message(FATAL_ERROR "PULSATRIX_ENABLE_PYTHON requires -DPYTHON_EXECUTABLE=<path> "
                             "(or PULSATRIX_PYTHON_EXECUTABLE env var) pointing at a Python "
                             "install with dev headers -- do not rely on PATH resolution, "
                             "multiple interpreters may be present.")
    endif()

    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)

    if(PULSATRIX_ENABLE_HIP AND NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION)
        # pybind11 adds -flto unless this variable is defined. With HIP on, pulsatrix_core
        # carries HIP sources, so the module is linked by ROCm's clang -- which cannot read
        # the GCC LTO bytecode in the bindings object and silently emits a .so with no
        # PyInit_pulsatrix_py ("dynamic module does not define module export function").
        # Defining it (OFF, CMake's own default) stops pybind11 adding the flag.
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
    endif()
    pybind11_add_module(pulsatrix_py bindings/pulsatrix_py.cpp)
    # pulsatrix_py depends ON pulsatrix_core; pulsatrix_core has zero dependency edge back onto this
    # target or any pybind11 header -- non-negotiable #2.
    target_link_libraries(pulsatrix_py PRIVATE pulsatrix_core)
    # pulsatrix_core is static and linked into a shared module. GNU ld tolerates the
    # distro default PIE objects; ROCm's lld (the linker once HIP is on) rejects them with
    # "relocation R_X86_64_PC32 cannot be used against symbol ...; recompile with -fPIC".
    set_target_properties(pulsatrix_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
    # Lets the bindings expose the GPU backends this build actually has (Tensor.to,
    # compiled_devices) without any core header learning about the binding layer.
    if(PULSATRIX_ENABLE_CUDA)
        target_compile_definitions(pulsatrix_py PRIVATE PULSATRIX_PY_WITH_CUDA)
    endif()
    if(PULSATRIX_ENABLE_HIP)
        target_compile_definitions(pulsatrix_py PRIVATE PULSATRIX_PY_WITH_HIP)
    endif()
endif()

# pulsatrix_viz's sources, listed outside the option so the install rules below can leave
# its headers out of the core package whether or not this build has the viz module.
set(PULSATRIX_VIZ_SOURCES
    src/viz/window.cpp
    src/viz/attribution_bar_chart.cpp
    src/viz/attribution_waterfall.cpp
    src/viz/attribution_beeswarm.cpp
    src/viz/saliency_heatmap_view.cpp
    src/viz/circuit_graph_view.cpp
    src/viz/confidence_meter.cpp
    src/viz/explanation_score_card.cpp
    src/viz/implot_metrics_sink_draw.cpp
    src/viz/training_dashboard.cpp
    src/viz/texture_cache.cpp
    src/viz/image_grid_view.cpp
    src/viz/dataset_statistics_view.cpp
    src/viz/token_relevance_view.cpp
    src/viz/fonts.cpp
)

if(PULSATRIX_ENABLE_VIZ)
    # Native C++ visualization layer (Dear ImGui + ImPlot) -- supersedes the charter's
    # original Python+matplotlib Phase 5 Mission 2 plan (see
    # plans/okay-we-have-now-buzzing-moth.md): live dashboards/scorecards stay in C++, no
    # Python runtime dependency. pulsatrix_viz depends on pulsatrix_core; pulsatrix_core has
    # zero dependency edge back onto this target or any ImGui/ImPlot/GLFW header -- same
    # one-directional discipline non-negotiable #2 already requires of pulsatrix_py.
    include(FetchContent)

    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.4
    )
    FetchContent_MakeAvailable(glfw)

    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG v1.92.5
    )
    FetchContent_MakeAvailable(imgui)

    FetchContent_Declare(
        implot
        GIT_REPOSITORY https://github.com/epezent/implot.git
        GIT_TAG v0.17
    )
    FetchContent_MakeAvailable(implot)

    find_package(OpenGL REQUIRED)

    # ImGui/ImPlot ship no CMakeLists.txt of their own -- wrap their sources into one static
    # library each, the same way this project already vendors GoogleTest/pybind11/stb.
    add_library(imgui STATIC
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
    )
    target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
    target_link_libraries(imgui PUBLIC glfw OpenGL::GL)
    # Every script's characters: 32-bit ImWchar reaches Unicode planes 1-16 (emoji, math
    # alphanumerics, historic scripts). PUBLIC: everything that includes imgui.h must agree.
    target_compile_definitions(imgui PUBLIC IMGUI_USE_WCHAR32)
    # FreeType, when installed, draws color emoji and hints small text better than stb_truetype.
    find_package(Freetype QUIET)
    option(PULSATRIX_VIZ_FREETYPE "Draw ImGui text with FreeType (color emoji) when it is installed" ON)
    if(PULSATRIX_VIZ_FREETYPE AND Freetype_FOUND)
        target_sources(imgui PRIVATE ${imgui_SOURCE_DIR}/misc/freetype/imgui_freetype.cpp)
        target_compile_definitions(imgui PUBLIC IMGUI_ENABLE_FREETYPE)
        target_link_libraries(imgui PUBLIC Freetype::Freetype)
        message(STATUS "pulsatrix_viz: ImGui text through FreeType ${FREETYPE_VERSION_STRING}")
    endif()

    add_library(implot STATIC
        ${implot_SOURCE_DIR}/implot.cpp
        ${implot_SOURCE_DIR}/implot_items.cpp
        ${implot_SOURCE_DIR}/implot_demo.cpp
    )
    target_include_directories(implot PUBLIC ${implot_SOURCE_DIR})
    target_link_libraries(implot PUBLIC imgui)

    # Phase A: explanation charts + score card, on top of Phase 0's VizWindow scaffolding.
    # Later phases (B: training dashboards, C: data/inference previews) add more sources here.
    add_library(pulsatrix_viz ${PULSATRIX_VIZ_SOURCES})
    target_include_directories(pulsatrix_viz PUBLIC include)
    # PUBLIC, not PRIVATE: demo executables (examples/viz/*.cpp) call ImGui/ImPlot directly
    # in their own translation units, so they need imgui/implot's include dirs propagated
    # transitively through pulsatrix_viz, not just linked into it.
    target_link_libraries(pulsatrix_viz PUBLIC pulsatrix_core imgui implot)
    # On Linux, fontconfig finds a fallback font for every script (viz/fonts.hpp).
    find_package(Fontconfig QUIET)
    if(Fontconfig_FOUND)
        target_link_libraries(pulsatrix_viz PRIVATE Fontconfig::Fontconfig)
        target_compile_definitions(pulsatrix_viz PRIVATE PULSATRIX_VIZ_WITH_FONTCONFIG)
    endif()

    add_executable(explanation_dashboard_demo examples/viz/explanation_dashboard_demo.cpp)
    target_link_libraries(explanation_dashboard_demo PRIVATE pulsatrix_viz)

    add_executable(training_dashboard_demo examples/viz/training_dashboard_demo.cpp)
    target_link_libraries(training_dashboard_demo PRIVATE pulsatrix_viz)

    add_executable(dataset_preview_demo examples/viz/dataset_preview_demo.cpp)
    target_link_libraries(dataset_preview_demo PRIVATE pulsatrix_viz)

    add_executable(live_inference_demo examples/viz/live_inference_demo.cpp)
    target_link_libraries(live_inference_demo PRIVATE pulsatrix_viz)

    # Every viz widget against real MNIST test digits (needs data/MNIST/raw, see
    # tools/fetch_mnist.py). --screenshot writes PNGs through the stb_image_write header the
    # already-fetched stb dependency ships (no new dependency).
    # Token relevance documents (VIZ-6a) in TokenRelevanceView; --screenshot writes a PNG.
    add_executable(token_relevance_demo examples/viz/token_relevance_demo.cpp)
    target_link_libraries(token_relevance_demo PRIVATE pulsatrix_viz)
    target_include_directories(token_relevance_demo SYSTEM PRIVATE ${stb_SOURCE_DIR})

    add_executable(mnist_viz_gallery examples/viz/mnist_viz_gallery.cpp)
    target_link_libraries(mnist_viz_gallery PRIVATE pulsatrix_viz)
    target_include_directories(mnist_viz_gallery SYSTEM PRIVATE ${stb_SOURCE_DIR})
endif()

find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    set(DOXYGEN_EXTRACT_ALL YES)
    set(DOXYGEN_WARN_IF_UNDOCUMENTED YES)
    set(DOXYGEN_WARN_NO_PARAMDOC YES)
    # doxygen-awesome-css (vendored, docs/doxygen-awesome/, v2.5.0, MIT) replaces Doxygen's
    # default theme with a clean, modern light/dark layout that actually matches the MkDocs
    # Material site's look (the earlier HTML_COLORSTYLE_HUE-only retint kept Doxygen's own
    # default template, which reads as a completely different, dated site next to Material).
    # pulsatrix-brand.css overrides its CSS custom properties to the same palette
    # docs/stylesheets/extra.css uses for the MkDocs site. HTML_COLORSTYLE=LIGHT disables
    # Doxygen's own separate built-in light/dark toggle (added ~1.9.5), since doxygen-awesome
    # already handles theming via `prefers-color-scheme` -- the two would otherwise conflict.
    # #projectlogo's own CSS caps the logo's display height regardless of the source image's
    # actual resolution -- the default theme has no such rule, which is why the logo rendered
    # far too large before this switch.
    set(DOXYGEN_PROJECT_LOGO ${CMAKE_SOURCE_DIR}/docs/assets/pulsatrix_icon.png)
    set(DOXYGEN_HTML_COLORSTYLE LIGHT)
    set(DOXYGEN_GENERATE_TREEVIEW YES)
    set(DOXYGEN_DISABLE_INDEX NO)
    set(DOXYGEN_FULL_SIDEBAR NO)
    set(DOXYGEN_HTML_EXTRA_STYLESHEET
        ${CMAKE_SOURCE_DIR}/docs/doxygen-awesome/doxygen-awesome.css
        ${CMAKE_SOURCE_DIR}/docs/doxygen-awesome/doxygen-awesome-sidebar-only.css
        ${CMAKE_SOURCE_DIR}/docs/doxygen-awesome/pulsatrix-brand.css
    )
    doxygen_add_docs(docs
        ${CMAKE_SOURCE_DIR}/include
        COMMENT "Generating pulsatrix API documentation"
    )
else()
    message(STATUS "Doxygen not found -- 'docs' target unavailable")
endif()

if(PULSATRIX_BUILD_EXAMPLES)
    add_executable(xor_demo examples/xor_demo.cpp)
    target_link_libraries(xor_demo PRIVATE pulsatrix_core)

    add_executable(neuro_symbolic_toy_kb_demo examples/neuro_symbolic_toy_kb_demo.cpp)
    target_link_libraries(neuro_symbolic_toy_kb_demo PRIVATE pulsatrix_core)

    add_executable(explainer_demo examples/explainer_demo.cpp)
    target_link_libraries(explainer_demo PRIVATE pulsatrix_core)

    add_executable(grad_cam_mnist_demo examples/grad_cam_mnist_demo.cpp)
    target_link_libraries(grad_cam_mnist_demo PRIVATE pulsatrix_core)

    add_executable(mnist_training_demo examples/mnist_training_demo.cpp)
    target_link_libraries(mnist_training_demo PRIVATE pulsatrix_core)

    add_executable(mnist_dataloader_demo examples/mnist_dataloader_demo.cpp)
    target_link_libraries(mnist_dataloader_demo PRIVATE pulsatrix_core)

    add_executable(sequence_model_demo examples/sequence_model_demo.cpp)
    target_link_libraries(sequence_model_demo PRIVATE pulsatrix_core)

    add_executable(transformer_block_demo examples/transformer_block_demo.cpp)
    target_link_libraries(transformer_block_demo PRIVATE pulsatrix_core)

    add_executable(dqn_cartpole_demo examples/dqn_cartpole_demo.cpp)
    target_link_libraries(dqn_cartpole_demo PRIVATE pulsatrix_core)

    add_executable(reinforce_cartpole_demo examples/reinforce_cartpole_demo.cpp)
    target_link_libraries(reinforce_cartpole_demo PRIVATE pulsatrix_core)

    add_executable(a2c_cartpole_demo examples/a2c_cartpole_demo.cpp)
    target_link_libraries(a2c_cartpole_demo PRIVATE pulsatrix_core)

    add_executable(ppo_cartpole_demo examples/ppo_cartpole_demo.cpp)
    target_link_libraries(ppo_cartpole_demo PRIVATE pulsatrix_core)

    add_executable(sac_continuous_cartpole_demo examples/sac_continuous_cartpole_demo.cpp)
    target_link_libraries(sac_continuous_cartpole_demo PRIVATE pulsatrix_core)

    # Runs its gemm workload on the GPU backend this build has (HIP, else CUDA, else CPU).
    add_executable(system_monitor_demo examples/system_monitor_demo.cpp)
    target_link_libraries(system_monitor_demo PRIVATE pulsatrix_core)
    if(PULSATRIX_ENABLE_HIP)
        target_compile_definitions(system_monitor_demo PRIVATE PULSATRIX_DEMO_WITH_HIP)
    elseif(PULSATRIX_ENABLE_CUDA)
        target_compile_definitions(system_monitor_demo PRIVATE PULSATRIX_DEMO_WITH_CUDA)
    endif()

    # Recipes (docs campaign) -- small, didactic, one-tool-at-a-time programs, each paired
    # with a docs/recipes/<section>/<name>.md page. See docs/recipes/index.md.
    add_executable(xor_training_recipe examples/recipes/xor_training.cpp)
    target_link_libraries(xor_training_recipe PRIVATE pulsatrix_core)

    add_executable(tagger_finetune_recipe examples/recipes/tagger_finetune.cpp)
    target_link_libraries(tagger_finetune_recipe PRIVATE pulsatrix_core)

    add_executable(csv_dataloader_training_recipe examples/recipes/csv_dataloader_training.cpp)
    target_link_libraries(csv_dataloader_training_recipe PRIVATE pulsatrix_core)

    add_executable(sequence_models_recipe examples/recipes/sequence_models_rnn_lstm_gru.cpp)
    target_link_libraries(sequence_models_recipe PRIVATE pulsatrix_core)

    add_executable(residual_and_norm_layers_recipe examples/recipes/residual_and_norm_layers.cpp)
    target_link_libraries(residual_and_norm_layers_recipe PRIVATE pulsatrix_core)

    add_executable(kernel_shap_basics_recipe examples/recipes/kernel_shap_basics.cpp)
    target_link_libraries(kernel_shap_basics_recipe PRIVATE pulsatrix_core)

    add_executable(lime_basics_recipe examples/recipes/lime_basics.cpp)
    target_link_libraries(lime_basics_recipe PRIVATE pulsatrix_core)

    add_executable(saliency_and_ig_recipe examples/recipes/saliency_and_integrated_gradients.cpp)
    target_link_libraries(saliency_and_ig_recipe PRIVATE pulsatrix_core)

    add_executable(grad_cam_walkthrough_recipe examples/recipes/grad_cam_walkthrough.cpp)
    target_link_libraries(grad_cam_walkthrough_recipe PRIVATE pulsatrix_core)

    add_executable(dqn_cartpole_recipe examples/recipes/dqn_cartpole.cpp)
    target_include_directories(dqn_cartpole_recipe PRIVATE examples)
    target_link_libraries(dqn_cartpole_recipe PRIVATE pulsatrix_core)

    add_executable(gae_and_ppo_clipped_recipe examples/recipes/gae_and_ppo_clipped.cpp)
    target_link_libraries(gae_and_ppo_clipped_recipe PRIVATE pulsatrix_core)

    add_executable(sparse_autoencoder_probe_recipe examples/recipes/sparse_autoencoder_probe.cpp)
    target_link_libraries(sparse_autoencoder_probe_recipe PRIVATE pulsatrix_core)

    add_executable(gflownet_hypergrid_recipe examples/recipes/gflownet_hypergrid.cpp)
    target_link_libraries(gflownet_hypergrid_recipe PRIVATE pulsatrix_core)

    add_executable(datalog_lrp_bridge_recipe examples/recipes/datalog_lrp_bridge.cpp)
    target_link_libraries(datalog_lrp_bridge_recipe PRIVATE pulsatrix_core)

    add_executable(mnist_lrp_recipe examples/recipes/mnist_lrp.cpp)
    target_link_libraries(mnist_lrp_recipe PRIVATE pulsatrix_core)

    add_executable(protein_tem1_recipe examples/recipes/protein_tem1.cpp)
    target_link_libraries(protein_tem1_recipe PRIVATE pulsatrix_core)
    target_compile_definitions(protein_tem1_recipe PRIVATE $<$<BOOL:${PULSATRIX_ENABLE_HIP}>:PULSATRIX_DEMO_WITH_HIP>)
endif()

if(PULSATRIX_BUILD_TESTS)
    include(FetchContent)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
    )
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    add_subdirectory(tests)
endif()

# pulsatrix_svg: JSON viz documents to SVG figures from the command line (VIZ-2).
add_executable(pulsatrix_svg tools/render/pulsatrix_svg.cpp)
target_link_libraries(pulsatrix_svg PRIVATE pulsatrix_core)

# pulsatrix_golden: a Hugging Face model's logits against transformers' on real text (LLM-6).
add_executable(pulsatrix_golden tools/golden/pulsatrix_golden.cpp)
target_link_libraries(pulsatrix_golden PRIVATE pulsatrix_core)
target_compile_definitions(pulsatrix_golden PRIVATE $<$<BOOL:${PULSATRIX_ENABLE_HIP}>:PULSATRIX_GOLDEN_WITH_HIP>)

# pulsatrix_proteingym: ProteinGym zero-shot variant-effect benchmarks with an ESM-2 model (PLM-3).
add_executable(pulsatrix_proteingym tools/plm/pulsatrix_proteingym.cpp)
target_link_libraries(pulsatrix_proteingym PRIVATE pulsatrix_core)
target_compile_definitions(pulsatrix_proteingym PRIVATE $<$<BOOL:${PULSATRIX_ENABLE_HIP}>:PULSATRIX_GOLDEN_WITH_HIP>)

# pulsatrix_contacts: contacts from an ESM-2 model's attention, scored against structures (PLM-4).
add_executable(pulsatrix_contacts tools/plm/pulsatrix_contacts.cpp)
target_link_libraries(pulsatrix_contacts PRIVATE pulsatrix_core)
target_compile_definitions(pulsatrix_contacts PRIVATE $<$<BOOL:${PULSATRIX_ENABLE_HIP}>:PULSATRIX_GOLDEN_WITH_HIP>)

# pulsatrix_protein_views: an ESM-2 model's mutation map, logo, contacts and tracks for one protein (PLM-5).
add_executable(pulsatrix_protein_views tools/plm/pulsatrix_protein_views.cpp)
target_link_libraries(pulsatrix_protein_views PRIVATE pulsatrix_core)
target_compile_definitions(pulsatrix_protein_views PRIVATE $<$<BOOL:${PULSATRIX_ENABLE_HIP}>:PULSATRIX_GOLDEN_WITH_HIP>)

# pulsatrix_explain_protein: AttnLRP on an ESM-2 model, residue by residue, with its checks (PLM-6).
add_executable(pulsatrix_explain_protein tools/plm/pulsatrix_explain_protein.cpp)
target_link_libraries(pulsatrix_explain_protein PRIVATE pulsatrix_core)
target_compile_definitions(pulsatrix_explain_protein PRIVATE $<$<BOOL:${PULSATRIX_ENABLE_HIP}>:PULSATRIX_GOLDEN_WITH_HIP>)

# pulsatrix_train_esm: masked-LM training and evaluation of ESM-2-shaped models (PLM-7).
add_executable(pulsatrix_train_esm tools/plm/pulsatrix_train_esm.cpp)
target_link_libraries(pulsatrix_train_esm PRIVATE pulsatrix_core)
target_compile_definitions(pulsatrix_train_esm PRIVATE $<$<BOOL:${PULSATRIX_ENABLE_HIP}>:PULSATRIX_GOLDEN_WITH_HIP>)

# pulsatrix_finetune_esm: fine-tuning and probing ESM-2 with SequenceHead (PLM-7).
add_executable(pulsatrix_finetune_esm tools/plm/pulsatrix_finetune_esm.cpp)
target_link_libraries(pulsatrix_finetune_esm PRIVATE pulsatrix_core)
target_compile_definitions(pulsatrix_finetune_esm PRIVATE $<$<BOOL:${PULSATRIX_ENABLE_HIP}>:PULSATRIX_GOLDEN_WITH_HIP>)

# pulsatrix_probe_esm: probes and SAE features of ESM-2 residue representations (PLM-8).
add_executable(pulsatrix_probe_esm tools/plm/pulsatrix_probe_esm.cpp)
target_link_libraries(pulsatrix_probe_esm PRIVATE pulsatrix_core)
target_compile_definitions(pulsatrix_probe_esm PRIVATE $<$<BOOL:${PULSATRIX_ENABLE_HIP}>:PULSATRIX_GOLDEN_WITH_HIP>)

# pulsatrix_explain_text: AttnLRP on a language model's prediction, as a token relevance figure (VIZ-6a).
add_executable(pulsatrix_explain_text tools/explain/pulsatrix_explain_text.cpp)
target_link_libraries(pulsatrix_explain_text PRIVATE pulsatrix_core)
target_compile_definitions(pulsatrix_explain_text PRIVATE $<$<BOOL:${PULSATRIX_ENABLE_HIP}>:PULSATRIX_GOLDEN_WITH_HIP>)

# pulsatrix_tokenizer_parity: a tokenizer.json against Hugging Face tokenizers on a corpus (TOK-2).
add_executable(pulsatrix_tokenizer_parity tools/tokenizers/pulsatrix_tokenizer_parity.cpp)
target_link_libraries(pulsatrix_tokenizer_parity PRIVATE pulsatrix_core)

# pulsatrix_attnlrp: AttnLRP relevance on a Hugging Face model against LXT's (LLM-7).
add_executable(pulsatrix_attnlrp tools/golden/pulsatrix_attnlrp.cpp)
target_link_libraries(pulsatrix_attnlrp PRIVATE pulsatrix_core)
target_compile_definitions(pulsatrix_attnlrp PRIVATE $<$<BOOL:${PULSATRIX_ENABLE_HIP}>:PULSATRIX_GOLDEN_WITH_HIP>)

# pulsatrix_bench: the benchmark suite (KS-2), on every backend this build has.
add_executable(pulsatrix_bench tools/bench/pulsatrix_bench.cpp)
target_link_libraries(pulsatrix_bench PRIVATE pulsatrix_core)
target_compile_definitions(pulsatrix_bench PRIVATE
    PULSATRIX_BENCH_VERSION="${PROJECT_VERSION}"
    PULSATRIX_BENCH_BUILD_TYPE="$<IF:$<BOOL:$<CONFIG>>,$<CONFIG>,unspecified>"
    $<$<BOOL:${PULSATRIX_ENABLE_HIP}>:PULSATRIX_BENCH_WITH_HIP>
    $<$<BOOL:${PULSATRIX_ENABLE_CUDA}>:PULSATRIX_BENCH_WITH_CUDA>)

# Fixed GPU training workloads for scripts/profile_hip.sh (HIP-1).
if(PULSATRIX_ENABLE_HIP)
    add_executable(hip_profile_workloads tools/profile/hip_profile_workloads.cpp)
    target_link_libraries(hip_profile_workloads PRIVATE pulsatrix_core)
    target_include_directories(hip_profile_workloads PRIVATE tools/bench)
endif()

if(PULSATRIX_BUILD_FUZZERS)
    foreach(fuzzer safetensors_fuzz viz_document_fuzz)
        add_executable(${fuzzer} tools/fuzz/${fuzzer}.cpp)
        target_link_libraries(${fuzzer} PRIVATE pulsatrix_core)
        if(PULSATRIX_FUZZ_STANDALONE)
            target_compile_definitions(${fuzzer} PRIVATE PULSATRIX_FUZZ_STANDALONE)
        else()
            target_link_options(${fuzzer} PRIVATE -fsanitize=fuzzer)
        endif()
    endforeach()
endif()

# KS-1: `cmake --install` and find_package(pulsatrix). Installs pulsatrix_core as
# pulsatrix::core with its headers and a package config that finds the same dependencies
# (threads, and the CUDA or ROCm libraries when this build has those backends). pulsatrix_viz
# and the Python module aren't installed: viz would also have to install its fetched
# ImGui, ImPlot and GLFW, and the Python module is loaded from the build tree.
if(PULSATRIX_INSTALL)
    include(CMakePackageConfigHelpers)
    set(pulsatrix_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/pulsatrix")

    set(excluded_headers "")
    foreach(source IN LISTS PULSATRIX_VIZ_SOURCES)
        get_filename_component(name "${source}" NAME_WE)
        list(APPEND excluded_headers PATTERN "viz/${name}.hpp" EXCLUDE)
    endforeach()
    # A backend's headers include its SDK's, so they ship only with that backend.
    if(NOT PULSATRIX_ENABLE_CUDA)
        foreach(header cuda_backend cuda_check cublas_check)
            list(APPEND excluded_headers PATTERN "${header}.hpp" EXCLUDE)
        endforeach()
    endif()
    if(NOT PULSATRIX_ENABLE_HIP)
        foreach(header hip_backend hip_check hipblas_check)
            list(APPEND excluded_headers PATTERN "${header}.hpp" EXCLUDE)
        endforeach()
    endif()
    install(DIRECTORY include/pulsatrix
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        FILES_MATCHING PATTERN "*.hpp"
        ${excluded_headers})

    install(TARGETS pulsatrix_svg pulsatrix_bench pulsatrix_golden pulsatrix_attnlrp pulsatrix_tokenizer_parity pulsatrix_explain_text pulsatrix_proteingym pulsatrix_contacts pulsatrix_protein_views pulsatrix_explain_protein pulsatrix_train_esm pulsatrix_finetune_esm pulsatrix_probe_esm RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
    install(TARGETS pulsatrix_core
        EXPORT pulsatrixTargets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
    install(EXPORT pulsatrixTargets
        NAMESPACE pulsatrix::
        DESTINATION "${pulsatrix_cmake_dir}")

    configure_package_config_file(cmake/pulsatrixConfig.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/pulsatrixConfig.cmake"
        INSTALL_DESTINATION "${pulsatrix_cmake_dir}"
        PATH_VARS CMAKE_INSTALL_INCLUDEDIR)
    # Within a major version the API may still change (README, Status), but a 1.x build
    # satisfies find_package(pulsatrix 1.0).
    write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/pulsatrixConfigVersion.cmake"
        COMPATIBILITY SameMajorVersion)
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/pulsatrixConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/pulsatrixConfigVersion.cmake"
        DESTINATION "${pulsatrix_cmake_dir}")
endif()
