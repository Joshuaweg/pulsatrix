cmake_minimum_required(VERSION 3.20)
project(exai_dl_library CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(EXAI_ENABLE_CUDA "Build the CUDA DeviceBackend" OFF)
option(EXAI_ENABLE_PYTHON "Build Python bindings via pybind11" OFF)
option(EXAI_BUILD_TESTS "Build the GoogleTest suite" ON)

add_library(exai_core
    src/cpu_backend.cpp
)
target_include_directories(exai_core PUBLIC include)
target_compile_features(exai_core PUBLIC cxx_std_17)

if(EXAI_ENABLE_CUDA)
    message(STATUS "EXAI_ENABLE_CUDA is ON but no CUDA backend exists yet (Phase 1.5) — no-op for now")
endif()

if(EXAI_ENABLE_PYTHON)
    message(STATUS "EXAI_ENABLE_PYTHON is ON but no bindings exist yet (Phase 5) — no-op for now")
endif()

if(EXAI_BUILD_TESTS)
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
