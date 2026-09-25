# Phase 5 Mission 1, Objective 1: Attribution + ExplainerContext bindings.
# ExplainerContext's constructor validation (empty / null-containing module vector) is
# proven correct in tests/explainer_context_test.cpp's ConstructorThrowsOnEmptyModuleVector
# / ConstructorThrowsOnNullModulePointer -- these tests confirm the same std::invalid_argument
# boundary is reachable and correctly mapped to a Python ValueError through pybind11, not
# re-deriving the validation itself. Explainer-producing tests (Saliency/IntegratedGradients/
# GradCAM/LIME/KernelSHAP/PDP) land in this same file under Objectives 2-3.
import pytest

import pulsatrix_py


def test_explainer_context_constructs_from_module_chain():
    linear = pulsatrix_py.LinearModule(2, 2)
    relu = pulsatrix_py.ReluModule()

    # Must not raise -- a valid, non-empty module chain.
    pulsatrix_py.ExplainerContext([linear, relu])


def test_explainer_context_rejects_empty_module_list():
    with pytest.raises(ValueError):
        pulsatrix_py.ExplainerContext([])


def test_attribution_exposes_method_values_metadata():
    values = pulsatrix_py.Tensor.from_values([2], [1.0, 2.0])
    attribution = pulsatrix_py.Attribution("saliency", values, {"target_index": "0"})

    assert attribution.method == "saliency"
    assert attribution.values.at([0]) == 1.0
    assert attribution.values.at([1]) == 2.0
    assert attribution.metadata == {"target_index": "0"}


# Objective 2: graph-native explainers (Saliency, IntegratedGradients, GradCAM). Every
# fixture below is copied exactly from its C++ counterpart (saliency_test.cpp,
# integrated_gradients_test.cpp, grad_cam_test.cpp) -- reused, not re-derived, per this
# project's standing discipline.


def test_saliency_gradient_matches_weight_column_for_linear_only_network():
    # (in=3, out=2), row-major. d(output[1])/d(input[i]) = weight[i][1] exactly; bias set
    # large to prove it has zero effect on the gradient.
    linear = pulsatrix_py.LinearModule(3, 2)
    linear.set_weight([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    linear.set_bias([100.0, 100.0])

    ctx = pulsatrix_py.ExplainerContext([linear])
    x = pulsatrix_py.Tensor.from_values([3], [1.0, 1.0, 1.0])

    saliency = pulsatrix_py.Saliency()
    attr = saliency.explain(ctx, x, 1)

    assert attr.method == "saliency"
    assert attr.values.at([0]) == pytest.approx(2.0)
    assert attr.values.at([1]) == pytest.approx(4.0)
    assert attr.values.at([2]) == pytest.approx(6.0)
    assert attr.metadata["target_index"] == "1"


def test_integrated_gradients_completeness_axiom_holds_within_tolerance():
    linear1 = pulsatrix_py.LinearModule(3, 4)
    linear1.set_weight([0.2, -0.4, 0.6, 0.1, -0.3, 0.5, 0.7, -0.2, 0.1, 0.4, -0.6, 0.3])
    linear1.set_bias([0.1, -0.1, 0.2, 0.0])

    relu = pulsatrix_py.ReluModule()

    linear2 = pulsatrix_py.LinearModule(4, 2)
    linear2.set_weight([0.5, -0.3, 0.2, 0.4, -0.1, 0.6, 0.3, -0.5])
    linear2.set_bias([0.05, -0.05])

    ctx = pulsatrix_py.ExplainerContext([linear1, relu, linear2])

    x = pulsatrix_py.Tensor.from_values([3], [0.5, -0.3, 1.2])
    baseline = pulsatrix_py.Tensor.from_values([3], [0.0, 0.0, 0.0])
    target_index = 0

    # ExplainerContext.forward_pass() is deliberately not bound (Objective 1's Recon) --
    # direct chained Module.forward() calls are proven equivalent to it (Phase 2 Mission 0).
    f_x = linear2.forward(relu.forward(linear1.forward(x))).at([target_index])
    f_baseline = linear2.forward(relu.forward(linear1.forward(baseline))).at([target_index])

    ig = pulsatrix_py.IntegratedGradients()
    attr = ig.explain(ctx, x, baseline, target_index, 200)

    assert attr.method == "integrated_gradients"
    assert attr.metadata["steps"] == "200"

    sum_ig = sum(attr.values.at([i]) for i in range(3))
    assert sum_ig == pytest.approx(f_x - f_baseline, abs=1e-3)


def test_grad_cam_computes_hand_derived_cam_for_simple_network():
    conv = pulsatrix_py.Conv2DModule(1, 2, 2, 2)
    conv.set_kernel([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    conv.set_bias([0.0, 0.0])

    relu = pulsatrix_py.ReluModule()
    flatten = pulsatrix_py.FlattenModule()

    linear = pulsatrix_py.LinearModule(8, 2)
    linear.set_weight([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0])
    linear.set_bias([0.0, 0.0])

    ctx = pulsatrix_py.ExplainerContext([conv, relu, flatten, linear])
    x = pulsatrix_py.Tensor.from_values([1, 3, 3], [1.0, 2.0, 3.0, 4.0, 0.0, 5.0, 6.0, 7.0, 8.0])

    gradcam = pulsatrix_py.GradCAM()
    attr = gradcam.explain(ctx, x, 0)

    assert attr.method == "grad_cam"
    assert attr.values.at([0, 0]) == pytest.approx(13.0)
    assert attr.values.at([0, 1]) == pytest.approx(13.0)
    assert attr.values.at([1, 0]) == pytest.approx(23.0)
    assert attr.values.at([1, 1]) == pytest.approx(32.0)
    assert attr.metadata["target_index"] == "0"


def test_grad_cam_raises_when_graph_has_no_conv_layer():
    linear = pulsatrix_py.LinearModule(2, 2)
    ctx = pulsatrix_py.ExplainerContext([linear])
    x = pulsatrix_py.Tensor.from_values([2], [1.0, 1.0])

    gradcam = pulsatrix_py.GradCAM()
    with pytest.raises(ValueError):
        gradcam.explain(ctx, x, 0)


def test_integrated_gradients_raises_on_zero_steps():
    linear = pulsatrix_py.LinearModule(2, 1)
    ctx = pulsatrix_py.ExplainerContext([linear])
    x = pulsatrix_py.Tensor.from_values([2], [1.0, 1.0])
    baseline = pulsatrix_py.Tensor.from_values([2], [0.0, 0.0])

    ig = pulsatrix_py.IntegratedGradients()
    with pytest.raises(ValueError):
        ig.explain(ctx, x, baseline, 0, 0)


# Objective 3: surrogate explainers (LIME, KernelSHAP, PDP). All graph-free -- predict is
# a plain Python callable (a direct chained Module.forward() call, since
# ExplainerContext.forward_pass() is deliberately not bound). Fixtures copied exactly from
# lime_test.cpp / kernel_shap_test.cpp / pdp_test.cpp.


def test_lime_recovers_exact_weight_column_for_linear_only_network():
    linear = pulsatrix_py.LinearModule(3, 2)
    linear.set_weight([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    linear.set_bias([100.0, 100.0])  # deliberately large/irrelevant

    predict = linear.forward
    x = pulsatrix_py.Tensor.from_values([3], [1.0, 1.0, 1.0])

    lime = pulsatrix_py.LIME()
    attr = lime.explain(predict, x, target_index=1, num_samples=300, sigma=1.0, l2_lambda=0.0, seed=42)

    assert attr.method == "lime"
    assert attr.values.at([0]) == pytest.approx(2.0, abs=1e-2)
    assert attr.values.at([1]) == pytest.approx(4.0, abs=1e-2)
    assert attr.values.at([2]) == pytest.approx(6.0, abs=1e-2)
    assert attr.metadata["target_index"] == "1"


def test_lime_raises_on_non_positive_num_samples():
    linear = pulsatrix_py.LinearModule(2, 1)
    x = pulsatrix_py.Tensor.from_values([2], [1.0, 1.0])

    lime = pulsatrix_py.LIME()
    with pytest.raises(ValueError):
        lime.explain(linear.forward, x, target_index=0, num_samples=0, sigma=1.0, l2_lambda=0.0, seed=42)


def test_lime_raises_on_non_positive_sigma():
    linear = pulsatrix_py.LinearModule(2, 1)
    x = pulsatrix_py.Tensor.from_values([2], [1.0, 1.0])

    lime = pulsatrix_py.LIME()
    with pytest.raises(ValueError):
        lime.explain(linear.forward, x, target_index=0, num_samples=50, sigma=0.0, l2_lambda=0.0, seed=42)


def test_kernel_shap_two_features_match_closed_form_shapley_values():
    linear = pulsatrix_py.LinearModule(2, 1)
    linear.set_weight([2.0, -3.0])
    linear.set_bias([0.0])

    predict = linear.forward
    x = pulsatrix_py.Tensor.from_values([2], [1.0, 2.0])
    baseline = pulsatrix_py.Tensor.from_values([2], [0.0, 0.0])

    shap = pulsatrix_py.KernelSHAP()
    attr = shap.explain(predict, x, baseline, target_index=0)

    assert attr.method == "kernel_shap"
    assert attr.values.at([0]) == pytest.approx(2.0 * 1.0, abs=1e-3)
    assert attr.values.at([1]) == pytest.approx(-3.0 * 2.0, abs=1e-3)


def test_kernel_shap_raises_on_mismatched_input_and_baseline_shapes():
    linear = pulsatrix_py.LinearModule(2, 1)
    x = pulsatrix_py.Tensor.from_values([2], [1.0, 1.0])
    mismatched_baseline = pulsatrix_py.Tensor.from_values([3], [0.0, 0.0, 0.0])

    shap = pulsatrix_py.KernelSHAP()
    with pytest.raises(ValueError):
        shap.explain(linear.forward, x, mismatched_baseline, target_index=0)


def test_pdp_curve_is_exactly_linear_with_true_feature_weight_as_slope():
    linear = pulsatrix_py.LinearModule(3, 1)
    linear.set_weight([2.0, -3.0, 5.0])
    linear.set_bias([100.0])

    predict = linear.forward
    background = [
        pulsatrix_py.Tensor.from_values([3], [1.0, 1.0, 1.0]),
        pulsatrix_py.Tensor.from_values([3], [5.0, -2.0, 0.0]),
        pulsatrix_py.Tensor.from_values([3], [-3.0, 4.0, 2.0]),
    ]

    pdp = pulsatrix_py.PDP()
    attr = pdp.explain(predict, background, feature_index=1, target_index=0, grid_min=-2.0, grid_max=4.0, grid_size=7)

    assert attr.method == "pdp"
    v0 = attr.values.at([0])
    v6 = attr.values.at([6])
    grid_step = (4.0 - (-2.0)) / (7 - 1)
    empirical_slope = (v6 - v0) / (6 * grid_step)
    assert empirical_slope == pytest.approx(-3.0, abs=1e-3)


def test_pdp_raises_on_empty_background():
    linear = pulsatrix_py.LinearModule(2, 1)

    pdp = pulsatrix_py.PDP()
    with pytest.raises(ValueError):
        pdp.explain(linear.forward, [], feature_index=0, target_index=0, grid_min=0.0, grid_max=1.0, grid_size=5)
