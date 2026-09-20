# Phase 5 Mission 0: binding-layer correctness (marshalling, zero-copy numpy interop) --
# a different concern from Tensor's own correctness, which the GoogleTest suite
# (tests/tensor_test.cpp) already covers. See
# pybind11_bindings/context_pybind11_project_binding_design.md's worked example this
# suite is modeled on.
import numpy as np
import pytest

import exai_py


def test_tensor_zeros_shape():
    t = exai_py.Tensor.zeros([2, 3])
    assert t.shape() == [2, 3]
    assert t.numel() == 6


def test_tensor_zeros_is_actually_zero():
    t = exai_py.Tensor.zeros([4])
    for i in range(4):
        assert t.at([i]) == 0.0


def test_tensor_from_values_row_major():
    t = exai_py.Tensor.from_values([2, 2], [1.0, 2.0, 3.0, 4.0])
    assert t.at([0, 0]) == 1.0
    assert t.at([0, 1]) == 2.0
    assert t.at([1, 0]) == 3.0
    assert t.at([1, 1]) == 4.0


def test_tensor_set_at_is_visible_via_at():
    t = exai_py.Tensor.zeros([3])
    t.set_at([1], 7.0)
    assert t.at([1]) == 7.0


def test_tensor_numpy_zero_copy():
    t = exai_py.Tensor.zeros([4])
    arr = np.asarray(t)
    assert arr.shape == (4,)
    arr[0] = 5.0
    # Mutation via the numpy view is visible on the C++ side -- proves zero-copy, not a
    # marshalled round-trip copy.
    assert t.at([0]) == 5.0


def test_tensor_numpy_multidim_shape_and_strides():
    t = exai_py.Tensor.from_values([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    arr = np.asarray(t)
    assert arr.shape == (2, 3)
    np.testing.assert_array_equal(arr, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))
