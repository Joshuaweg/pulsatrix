add_executable(pulsatrix_tests
    device_backend_test.cpp
    cpu_backend_test.cpp
    shape_test.cpp
    assert_test.cpp
    tensor_test.cpp
    dataset_test.cpp
    iterable_dataset_test.cpp
    sampler_test.cpp
    transform_test.cpp
    collate_test.cpp
    bounded_queue_test.cpp
    data_thread_pool_test.cpp
    data_loader_test.cpp
    csv_reader_test.cpp
    csv_dataset_test.cpp
    csv_dataloader_integration_test.cpp
    mnist_dataset_adapter_test.cpp
    image_decoder_test.cpp
    image_transforms_test.cpp
    image_folder_dataset_test.cpp
    dataloader_image_benchmark_test.cpp
    tokenizer_test.cpp
    vocabulary_test.cpp
    text_dataset_test.cpp
    text_collate_test.cpp
    text_dataloader_integration_test.cpp
    wav_reader_test.cpp
    audio_folder_dataset_test.cpp
    audio_transforms_test.cpp
    audio_collate_test.cpp
    audio_dataloader_integration_test.cpp
    video_frame_directory_dataset_test.cpp
    video_transforms_test.cpp
    dataset_validator_test.cpp
    search_space_test.cpp
    trial_test.cpp
    hpo_sampling_test.cpp
    hpo_xor_tuning_integration_test.cpp
    node_test.cpp
    computation_graph_test.cpp
    computation_graph_persistence_test.cpp
    autograd_test.cpp
    module_test.cpp
    module_graph_wiring_test.cpp
    linear_module_test.cpp
    relu_module_test.cpp
    softmax_module_test.cpp
    rope_module_test.cpp
    multihead_attention_module_test.cpp
    swiglu_module_test.cpp
    transformer_block_test.cpp
    residual_module_test.cpp
    conv2d_module_test.cpp
    rms_norm_module_test.cpp
    layer_norm_module_test.cpp
    group_norm_module_test.cpp
    batch_norm_module_test.cpp
    max_pool2d_module_test.cpp
    avg_pool2d_module_test.cpp
    embedding_module_test.cpp
    dropout_module_test.cpp
    sequential_module_test.cpp
    rnn_module_test.cpp
    lstm_module_test.cpp
    gru_module_test.cpp
    mamba_module_test.cpp
    rwkv_module_test.cpp
    retnet_module_test.cpp
    reparameterize_test.cpp
    noise_schedule_test.cpp
    sinusoidal_timestep_embedding_test.cpp
    cartpole_env_test.cpp
    continuous_cartpole_env_test.cpp
    hypergrid_env_test.cpp
    learnable_scalar_test.cpp
    gflownet_forward_policy_test.cpp
    trajectory_balance_loss_test.cpp
    detailed_balance_loss_test.cpp
    subtb_loss_test.cpp
    subtb_integration_test.cpp
    cross_objective_validation_test.cpp
    detailed_balance_integration_test.cpp
    gflownet_trajectory_test.cpp
    trajectory_balance_integration_test.cpp
    replay_buffer_test.cpp
    rollout_buffer_test.cpp
    dqn_loss_test.cpp
    dqn_target_test.cpp
    dqn_agent_test.cpp
    categorical_policy_agent_test.cpp
    policy_gradient_loss_test.cpp
    gae_test.cpp
    ppo_clipped_loss_test.cpp
    tanh_gaussian_policy_test.cpp
    polyak_update_test.cpp
    mse_loss_test.cpp
    calibration_loss_test.cpp
    cross_entropy_loss_test.cpp
    kl_divergence_loss_test.cpp
    bce_with_logits_loss_test.cpp
    vae_integration_test.cpp
    gan_integration_test.cpp
    diffusion_integration_test.cpp
    dqn_cartpole_integration_test.cpp
    reinforce_cartpole_integration_test.cpp
    a2c_cartpole_integration_test.cpp
    ppo_cartpole_integration_test.cpp
    sac_continuous_cartpole_integration_test.cpp
    mnist_loader_test.cpp
    mnist_classifier_example_test.cpp
    sgd_optimizer_test.cpp
    adam_optimizer_test.cpp
    metrics_sink_test.cpp
    xor_training_example_test.cpp
    attribution_test.cpp
    explainer_context_test.cpp
    activation_snapshot_test.cpp
    circuit_graph_test.cpp
    linear_probe_test.cpp
    sparse_autoencoder_test.cpp
    saliency_test.cpp
    integrated_gradients_test.cpp
    flatten_module_test.cpp
    grad_cam_test.cpp
    weighted_linear_regression_test.cpp
    linear_algebra_test.cpp
    gaussian_process_test.cpp
    acquisition_functions_test.cpp
    gp_bo_test.cpp
    gp_bo_xor_integration_test.cpp
    tpe_test.cpp
    tpe_xor_integration_test.cpp
    successive_halving_test.cpp
    successive_halving_xor_integration_test.cpp
    hyperband_test.cpp
    hyperband_xor_integration_test.cpp
    asha_test.cpp
    asha_xor_integration_test.cpp
    lime_test.cpp
    kernel_shap_test.cpp
    pdp_test.cpp
    lrp_conservation_test.cpp
    stability_test.cpp
    captum_benchmark_test.cpp
)

if(PULSATRIX_ENABLE_CUDA)
    target_sources(pulsatrix_tests PRIVATE cuda_check_test.cpp cuda_backend_test.cpp cublas_check_test.cpp
                                       backend_equivalence_test.cpp forward_pass_equivalence_test.cpp)
endif()

if(PULSATRIX_ENABLE_HIP)
    target_sources(pulsatrix_tests PRIVATE hip_check_test.cpp hip_backend_test.cpp hipblas_check_test.cpp
                                       backend_equivalence_hip_test.cpp forward_pass_equivalence_hip_test.cpp)
endif()

# The direct CUDA<->HIP arm of the charter's three-way gate. Compiled only when both vendor
# backends are on, which no machine available to this project can currently satisfy -- see
# the file's own header. Guarded rather than omitted so the gate is satisfiable unchanged the
# day a dual-vendor node exists.
if(PULSATRIX_ENABLE_CUDA AND PULSATRIX_ENABLE_HIP)
    target_sources(pulsatrix_tests PRIVATE three_way_equivalence_test.cpp)
endif()

target_link_libraries(pulsatrix_tests PRIVATE pulsatrix_core GTest::gtest_main GTest::gmock)

# ctest runs pulsatrix_tests with cwd = the build directory, not the source tree -- a relative
# "data/MNIST/raw/..." path never resolves there. mnist_loader_test.cpp needs an absolute
# path to the gitignored, fetch_mnist.py-populated data directory regardless of how/where
# ctest is invoked from.
target_compile_definitions(pulsatrix_tests PRIVATE PULSATRIX_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/data")

# Same rationale as PULSATRIX_TEST_DATA_DIR above, for small synthetic fixtures checked
# directly into the repo (not gitignored/downloaded like MNIST) -- csv_reader_test.cpp /
# csv_dataset_test.cpp (campaign_exai_dl_library_data_pipeline, Mission 3).
target_compile_definitions(pulsatrix_tests PRIVATE PULSATRIX_TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures")

include(GoogleTest)
# DISCOVERY_MODE PRE_TEST (CMake 3.18+): defers test discovery to ctest run-time instead of
# CMake's default POST_BUILD mode, which hardcodes whichever config's binary happened to be
# built most recently into the generated test commands -- under a multi-config generator
# (Visual Studio), that means `ctest -C Debug` can silently run the *Release* binary if
# Release was built more recently, with no error, just wrong results. PRE_TEST re-discovers
# against the actual $<CONFIG> binary on every ctest invocation.
gtest_discover_tests(pulsatrix_tests DISCOVERY_MODE PRE_TEST)
