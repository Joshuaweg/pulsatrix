cmake_minimum_required(VERSION 3.20)
project(pulsatrix CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(PULSATRIX_ENABLE_CUDA "Build the CUDA DeviceBackend" OFF)
option(PULSATRIX_ENABLE_HIP "Build the HIP/ROCm DeviceBackend" OFF)
option(PULSATRIX_ENABLE_PYTHON "Build Python bindings via pybind11" OFF)
option(PULSATRIX_BUILD_TESTS "Build the GoogleTest suite" ON)
option(PULSATRIX_BUILD_EXAMPLES "Build standalone example executables (examples/)" ON)

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
    src/residual_module.cpp
    src/conv2d_module.cpp
    src/rms_norm_module.cpp
    src/layer_norm_module.cpp
    src/group_norm_module.cpp
    src/batch_norm_module.cpp
    src/max_pool2d_module.cpp
    src/avg_pool2d_module.cpp
    src/embedding_module.cpp
    src/dropout_module.cpp
    src/sequential_module.cpp
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
)
target_include_directories(pulsatrix_core PUBLIC include)
target_compile_features(pulsatrix_core PUBLIC cxx_std_17)

# First consumer: DataThreadPool/BoundedQueue (campaign_exai_dl_library_data_pipeline,
# Mission 2) -- pulsatrix's first std::thread-based code, needs the platform thread
# library linked explicitly (not implicit via any other dependency).
find_package(Threads REQUIRED)
target_link_libraries(pulsatrix_core PUBLIC Threads::Threads)

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

    pybind11_add_module(pulsatrix_py bindings/pulsatrix_py.cpp)
    # pulsatrix_py depends ON pulsatrix_core; pulsatrix_core has zero dependency edge back onto this
    # target or any pybind11 header -- non-negotiable #2.
    target_link_libraries(pulsatrix_py PRIVATE pulsatrix_core)
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

    # Recipes (docs campaign) -- small, didactic, one-tool-at-a-time programs, each paired
    # with a docs/recipes/<section>/<name>.md page. See docs/recipes/index.md.
    add_executable(xor_training_recipe examples/recipes/xor_training.cpp)
    target_link_libraries(xor_training_recipe PRIVATE pulsatrix_core)

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
endif()

if(PULSATRIX_BUILD_TESTS)
    include(FetchContent)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
    )
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    add_subdirectory(tests)
endif()
