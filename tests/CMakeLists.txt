add_executable(pulsatrix_tests
    device_backend_test.cpp
    cpu_backend_test.cpp
    shape_test.cpp
    assert_test.cpp
    tensor_test.cpp
    backend_primitives_test.cpp
    dataset_test.cpp
    iterable_dataset_test.cpp
    sampler_test.cpp
    transform_test.cpp
    collate_test.cpp
    bounded_queue_test.cpp
    data_thread_pool_test.cpp
    system_monitor_test.cpp
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
    text_tokenizer_test.cpp
    tokenizer_json_test.cpp
    word_scores_test.cpp
    text_relevance_test.cpp
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
    individual_test.cpp
    selection_test.cpp
    crossover_test.cpp
    mutation_test.cpp
    survivor_selection_test.cpp
    evolutionary_loop_test.cpp
    nsga2_test.cpp
    search_space_test.cpp
    trial_test.cpp
    hpo_sampling_test.cpp
    hpo_xor_tuning_integration_test.cpp
    hpo_genotype_test.cpp
    ga_hpo_xor_integration_test.cpp
    cma_es_test.cpp
    cma_es_xor_integration_test.cpp
    neat_genome_test.cpp
    neat_phenotype_test.cpp
    neat_speciation_test.cpp
    neat_xor_fitness_test.cpp
    neat_evolution_test.cpp
    evolution_strategies_test.cpp
    fixed_topology_xor_network_test.cpp
    pbt_test.cpp
    pbt_xor_integration_test.cpp
    generator_population_test.cpp
    generator_population_gan_integration_test.cpp
    egan_mutation_test.cpp
    egan_training_test.cpp
    egan_training_integration_test.cpp
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
    attention_reference_test.cpp
    attention_llm_features_test.cpp
    tied_lm_head_module_test.cpp
    generation_test.cpp
    kv_cache_test.cpp
    hf_model_test.cpp
    causal_lm_test.cpp
    golden_logits_test.cpp
    attnlrp_parity_test.cpp
    swiglu_module_test.cpp
    transformer_block_test.cpp
    residual_module_test.cpp
    conjunction_module_test.cpp
    disjunction_module_test.cpp
    negation_module_test.cpp
    aggregator_module_test.cpp
    satisfaction_loss_test.cpp
    neuro_symbolic_toy_kb_test.cpp
    datalog_term_test.cpp
    datalog_atom_test.cpp
    datalog_rule_test.cpp
    datalog_fact_database_test.cpp
    datalog_engine_test.cpp
    datalog_semiring_test.cpp
    datalog_weighted_fact_database_test.cpp
    datalog_weighted_engine_test.cpp
    datalog_dual_semiring_test.cpp
    neuro_symbolic_datalog_bridge_test.cpp
    datalog_lrp_test.cpp
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
    named_parameters_test.cpp
    parameter_freezing_test.cpp
    top_k_test.cpp
    matrix_decompositions_test.cpp
    batch_norm_eval_test.cpp
    conv2d_stride_padding_test.cpp
    determinism_test.cpp
    device_consistency_test.cpp
    safetensors_test.cpp
    checkpoint_test.cpp
    param_groups_test.cpp
    adamw_sgd_momentum_test.cpp
    grad_clipping_test.cpp
    lr_scheduler_test.cpp
    token_cross_entropy_test.cpp
    tagger_finetune_test.cpp
    explanation_metrics_test.cpp
    null_model_baseline_test.cpp
    json_test.cpp
    viz_document_test.cpp
    svg_render_test.cpp
    html_render_test.cpp
    attribution_graph_test.cpp
    relevance_graph_test.cpp
    encoder_block_test.cpp
    encoder_lm_test.cpp
    variant_scoring_test.cpp
    protein_contacts_test.cpp
    protein_structure_test.cpp
    protein_views_test.cpp
    protein_explanations_test.cpp
    protein_training_test.cpp
    featurizer_test.cpp
    protein_concepts_test.cpp
    fitness_metrics_test.cpp
    benchmark_test.cpp
    caching_allocator_test.cpp
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
    lrp_test.cpp
    lrp_rules_test.cpp
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
    ice_test.cpp
    sensitivity_test.cpp
    counterfactual_test.cpp
    global_sensitivity_test.cpp
    lrp_conservation_test.cpp
    stability_test.cpp
    captum_benchmark_test.cpp
    lrp_reference_test.cpp
    colormap_test.cpp
    plot_data_test.cpp
    lrp_conservation_result_test.cpp
    explainer_stability_test.cpp
    implot_metrics_sink_test.cpp
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
# The tokenizer tests hold UTF-8 string literals (Cyrillic, CJK, emoji); MSVC otherwise reads
# sources in the system code page.
target_compile_options(pulsatrix_tests PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)

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

# KS-1: install this build, then build and run a separate project that finds it with
# find_package(pulsatrix) alone. Runs the install rules for real, so it catches a header the
# install leaves out, a dependency the package config doesn't find, and a broken export.
add_test(NAME PulsatrixInstall.FindPackageConsumer
    COMMAND "${CMAKE_COMMAND}"
        "-DBUILD_DIR=${CMAKE_BINARY_DIR}"
        "-DCONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/install_consumer"
        "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/install_consumer_work"
        "-DCONFIG=$<CONFIG>"
        "-DGENERATOR=${CMAKE_GENERATOR}"
        "-DCXX_COMPILER=${CMAKE_CXX_COMPILER}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/install_consumer_test.cmake")

# VIZ-2: the pulsatrix_svg command-line tool, on the VIZ-1 document fixtures.
add_test(NAME PulsatrixSvg.RendersAnAttribution
    COMMAND pulsatrix_svg "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/viz/attribution_features.v1.json" --top-k 3
        -o "${CMAKE_CURRENT_BINARY_DIR}/pulsatrix_svg_attribution.svg")
add_test(NAME PulsatrixSvg.RendersAHeatmap
    COMMAND pulsatrix_svg "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/viz/heatmap.v1.json"
        -o "${CMAKE_CURRENT_BINARY_DIR}/pulsatrix_svg_heatmap.svg")
add_test(NAME PulsatrixSvg.RendersCenteredIce
    COMMAND pulsatrix_svg "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/viz/partial_dependence.v1.json" --ice centered
        -o "${CMAKE_CURRENT_BINARY_DIR}/pulsatrix_svg_partial_dependence.svg")
add_test(NAME PulsatrixSvg.RendersATornado
    COMMAND pulsatrix_svg "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/viz/sensitivity.v1.json" --top-k 1
        -o "${CMAKE_CURRENT_BINARY_DIR}/pulsatrix_svg_tornado.svg")
add_test(NAME PulsatrixSvg.RendersACounterfactual
    COMMAND pulsatrix_svg "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/viz/counterfactual.v1.json"
        -o "${CMAKE_CURRENT_BINARY_DIR}/pulsatrix_svg_counterfactual.svg")
add_test(NAME PulsatrixSvg.RendersMorrisAndSobol
    COMMAND pulsatrix_svg "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/viz/sobol.v1.json"
        -o "${CMAKE_CURRENT_BINARY_DIR}/pulsatrix_svg_sobol.svg")
add_test(NAME PulsatrixSvg.RendersACounterfactualSet
    COMMAND pulsatrix_svg "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/viz/counterfactual.v1.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/viz/counterfactual.v1.json"
        -o "${CMAKE_CURRENT_BINARY_DIR}/pulsatrix_svg_counterfactual_set.svg")
# PLM-5: the protein documents, as figures and as pages.
foreach(kind mutation_map sequence_logo contact_map residue_tracks)
    add_test(NAME PulsatrixSvg.Renders_${kind}
        COMMAND pulsatrix_svg "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/viz/${kind}.v1.json"
            -o "${CMAKE_CURRENT_BINARY_DIR}/pulsatrix_svg_${kind}.svg")
    add_test(NAME PulsatrixSvg.Renders_${kind}_page
        COMMAND pulsatrix_svg "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/viz/${kind}.v1.json"
            -o "${CMAKE_CURRENT_BINARY_DIR}/pulsatrix_svg_${kind}.html")
endforeach()
add_test(NAME PulsatrixSvg.RejectsAnUnknownIceStyle
    COMMAND pulsatrix_svg "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/viz/partial_dependence.v1.json" --ice sideways)
set_tests_properties(PulsatrixSvg.RejectsAnUnknownIceStyle PROPERTIES WILL_FAIL TRUE)
add_test(NAME PulsatrixSvg.RejectsAKindWithNoView
    COMMAND pulsatrix_svg "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/viz/circuit_graph.v1.json")
set_tests_properties(PulsatrixSvg.RejectsAKindWithNoView PROPERTIES WILL_FAIL TRUE)
add_test(NAME PulsatrixSvg.RejectsSeveralDocumentsWithoutBeeswarm
    COMMAND pulsatrix_svg "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/viz/attribution.v1.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/viz/attribution.v1.json")
set_tests_properties(PulsatrixSvg.RejectsSeveralDocumentsWithoutBeeswarm PROPERTIES WILL_FAIL TRUE)
# VIZ-3: HTML output from the same tool.
add_test(NAME PulsatrixSvg.WritesHtml
    COMMAND pulsatrix_svg "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/viz/training_log.v1.json"
        -o "${CMAKE_CURRENT_BINARY_DIR}/pulsatrix_svg_training_log.html")
add_test(NAME PulsatrixSvg.RejectsATrainingLogAsSvg
    COMMAND pulsatrix_svg "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/viz/training_log.v1.json")
set_tests_properties(PulsatrixSvg.RejectsATrainingLogAsSvg PROPERTIES WILL_FAIL TRUE)
add_test(NAME PulsatrixSvg.InlineJsNeedsTheScripts
    COMMAND pulsatrix_svg "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/viz/heatmap.v1.json" --html
        --inline-js "${CMAKE_CURRENT_BINARY_DIR}/no_such_vega_dir")
set_tests_properties(PulsatrixSvg.InlineJsNeedsTheScripts PROPERTIES WILL_FAIL TRUE)
# VIZ-4: circuit graphs for Neuronpedia's viewer.
add_test(NAME PulsatrixSvg.ExportsACircuitGraphForNeuronpedia
    COMMAND pulsatrix_svg "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/viz/circuit_graph_finite.v1.json" --neuronpedia
        --slug xor --scan xor-mlp -o "${CMAKE_CURRENT_BINARY_DIR}/pulsatrix_svg_circuit_neuronpedia.json")
add_test(NAME PulsatrixSvg.NeuronpediaRefusesANonFiniteCircuit
    COMMAND pulsatrix_svg "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/viz/circuit_graph.v1.json" --neuronpedia)
set_tests_properties(PulsatrixSvg.NeuronpediaRefusesANonFiniteCircuit PROPERTIES WILL_FAIL TRUE)
add_test(NAME PulsatrixSvg.NeuronpediaNeedsACircuitGraph
    COMMAND pulsatrix_svg "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/viz/heatmap.v1.json" --neuronpedia)
set_tests_properties(PulsatrixSvg.NeuronpediaNeedsACircuitGraph PROPERTIES WILL_FAIL TRUE)
# LLM-6: the golden-logit tool, on a tiny model.
add_test(NAME PulsatrixGolden.TinyLlamaMatchesTransformers
    COMMAND pulsatrix_golden "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/hf_tiny/llama")
# TOK-2: the tokenizer parity tool, on a tiny tokenizer with the real gpt-oss pipeline.
add_test(NAME PulsatrixTokenizerParity.TinyGptOssMatchesHuggingFace
    COMMAND pulsatrix_tokenizer_parity "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tokenizers/gpt-oss/tokenizer.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tokenizers/gpt-oss/reference.jsonl")
# LLM-7: the AttnLRP parity tool, on a tiny model, and its negative control.
add_test(NAME PulsatrixAttnLrp.TinyLlamaMatchesLxt
    COMMAND pulsatrix_attnlrp "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/hf_tiny/llama")
add_test(NAME PulsatrixAttnLrp.TinyLlamaDoesNotMatchPlainGradient
    COMMAND pulsatrix_attnlrp "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/hf_tiny/llama"
        --reference "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/hf_tiny/llama/gradient_x_input.safetensors")
set_tests_properties(PulsatrixAttnLrp.TinyLlamaDoesNotMatchPlainGradient PROPERTIES WILL_FAIL TRUE)

# KS-2: LRP conservation on the benchmark models must hold on the CPU, whatever the timings.
add_test(NAME PulsatrixBench.CpuConservationCheck
    COMMAND pulsatrix_bench run --device cpu --quick --filter conservation --check)

# HIP-9: the host-kernel check behind scripts/rocm-build.sh (a bash script, so not on Windows).
if(UNIX)
    add_test(NAME CheckHostKernel.ClassifiesKernelReleases
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/check_host_kernel_test.sh")
endif()
