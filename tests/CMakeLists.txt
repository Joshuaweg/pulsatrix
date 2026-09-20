add_executable(exai_tests
    device_backend_test.cpp
    cpu_backend_test.cpp
    shape_test.cpp
    assert_test.cpp
    tensor_test.cpp
    node_test.cpp
    computation_graph_test.cpp
    computation_graph_persistence_test.cpp
    autograd_test.cpp
    module_test.cpp
    module_graph_wiring_test.cpp
    linear_module_test.cpp
    relu_module_test.cpp
    conv2d_module_test.cpp
    mse_loss_test.cpp
    sgd_optimizer_test.cpp
    adam_optimizer_test.cpp
    metrics_sink_test.cpp
    xor_training_example_test.cpp
    attribution_test.cpp
    explainer_context_test.cpp
    saliency_test.cpp
    integrated_gradients_test.cpp
    flatten_module_test.cpp
    grad_cam_test.cpp
    weighted_linear_regression_test.cpp
    lime_test.cpp
    kernel_shap_test.cpp
    pdp_test.cpp
    lrp_conservation_test.cpp
    stability_test.cpp
    captum_benchmark_test.cpp
)

if(EXAI_ENABLE_CUDA)
    target_sources(exai_tests PRIVATE cuda_check_test.cpp cuda_backend_test.cpp cublas_check_test.cpp
                                       backend_equivalence_test.cpp forward_pass_equivalence_test.cpp)
endif()

if(EXAI_ENABLE_HIP)
    target_sources(exai_tests PRIVATE hip_check_test.cpp hip_backend_test.cpp hipblas_check_test.cpp
                                       backend_equivalence_hip_test.cpp)
endif()

target_link_libraries(exai_tests PRIVATE exai_core GTest::gtest_main GTest::gmock)

include(GoogleTest)
# DISCOVERY_MODE PRE_TEST (CMake 3.18+): defers test discovery to ctest run-time instead of
# CMake's default POST_BUILD mode, which hardcodes whichever config's binary happened to be
# built most recently into the generated test commands -- under a multi-config generator
# (Visual Studio), that means `ctest -C Debug` can silently run the *Release* binary if
# Release was built more recently, with no error, just wrong results. PRE_TEST re-discovers
# against the actual $<CONFIG> binary on every ctest invocation.
gtest_discover_tests(exai_tests DISCOVERY_MODE PRE_TEST)
