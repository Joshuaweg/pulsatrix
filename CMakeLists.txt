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
    src/cross_entropy_loss.cpp
    src/kl_divergence_loss.cpp
    src/bce_with_logits_loss.cpp
    src/mnist_loader.cpp
    src/mnist_classifier_example.cpp
    src/sgd_optimizer.cpp
    src/adam_optimizer.cpp
    src/xor_training_example.cpp
)
target_include_directories(pulsatrix_core PUBLIC include)
target_compile_features(pulsatrix_core PUBLIC cxx_std_17)

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
