# Phase 5 Mission 0: Module bindings correctness -- construct a known network from
# Python, run forward(), compare against the exact same hand-verified fixture already
# proven correct in tests/linear_module_test.cpp's ForwardComputesHandVerifiedOutput
# (not a new derivation).
import exai_py


def test_linear_module_forward_matches_hand_verified_cpp_fixture():
    # in_features=2, out_features=2. W = [[1,2],[3,4]] (in x out, row-major), b = [0.5, -0.5].
    # x = [1, 1]. y = x @ W + b = [4.5, 5.5]  -- see linear_module_test.cpp
    linear = exai_py.LinearModule(2, 2)
    linear.set_weight([1.0, 2.0, 3.0, 4.0])
    linear.set_bias([0.5, -0.5])

    x = exai_py.Tensor.from_values([2], [1.0, 1.0])
    y = linear.forward(x)

    assert y.at([0]) == 4.5
    assert y.at([1]) == 5.5


def test_relu_module_forward_zeroes_negatives():
    relu = exai_py.ReluModule()
    x = exai_py.Tensor.from_values([3], [-1.0, 0.0, 2.0])
    y = relu.forward(x)

    assert y.at([0]) == 0.0
    assert y.at([1]) == 0.0
    assert y.at([2]) == 2.0


def test_conv2d_module_forward_matches_hand_verified_cpp_fixture():
    # Same fixture as conv2d_module_test.cpp's ForwardComputesHandVerifiedOutput:
    # 1 in-channel, 1 out-channel, 3x3 input, 2x2 kernel [[1,0],[0,1]], stride 1.
    # input = [[1,2,3],[4,5,6],[7,8,9]] -> output = [[6,8],[12,14]]
    conv = exai_py.Conv2DModule(1, 1, 2, 2)
    conv.set_kernel([1.0, 0.0, 0.0, 1.0])
    conv.set_bias([0.0])

    x = exai_py.Tensor.from_values([1, 3, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    y = conv.forward(x)

    assert y.at([0, 0, 0]) == 6.0
    assert y.at([0, 0, 1]) == 8.0
    assert y.at([0, 1, 0]) == 12.0
    assert y.at([0, 1, 1]) == 14.0
