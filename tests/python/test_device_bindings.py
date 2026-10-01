# Tensor.to / DeviceType bindings. The Cpu cases run on every build; the GPU cases run once
# per GPU device this build was compiled with, and are skipped otherwise -- the binding's
# own compiled_devices() decides, not the test.
import numpy as np
import pytest

import pulsatrix_py
from pulsatrix_py import DeviceType

GPU_DEVICES = [d for d in pulsatrix_py.compiled_devices() if d != DeviceType.Cpu]


def test_cpu_is_always_compiled():
    assert DeviceType.Cpu in pulsatrix_py.compiled_devices()


def test_new_tensor_is_on_cpu():
    assert pulsatrix_py.Tensor.zeros([2]).device() == DeviceType.Cpu


def test_to_cpu_on_cpu_tensor_is_noop_and_returns_self():
    t = pulsatrix_py.Tensor.from_values([2], [1.0, 2.0])
    assert t.to(DeviceType.Cpu) is t
    assert t.at([1]) == 2.0


@pytest.mark.parametrize("device", [DeviceType.Cuda, DeviceType.Hip])
def test_to_uncompiled_device_raises(device):
    if device in pulsatrix_py.compiled_devices():
        pytest.skip(f"{device} is compiled into this build")
    t = pulsatrix_py.Tensor.zeros([2])
    with pytest.raises(ValueError):
        t.to(device)
    assert t.device() == DeviceType.Cpu


@pytest.mark.parametrize("device", GPU_DEVICES)
def test_round_trip_through_gpu_preserves_values(device):
    t = pulsatrix_py.Tensor.from_values([2, 2], [1.5, -2.0, 0.0, 9.25])
    t.to(device)
    assert t.device() == device
    t.to(DeviceType.Cpu)
    np.testing.assert_array_equal(np.asarray(t), np.array([[1.5, -2.0], [0.0, 9.25]], dtype=np.float32))


@pytest.mark.parametrize("device", GPU_DEVICES)
def test_host_access_on_gpu_tensor_raises_instead_of_reading_device_memory(device):
    t = pulsatrix_py.Tensor.zeros([2]).to(device)
    with pytest.raises(ValueError):
        t.at([0])
    with pytest.raises(ValueError):
        t.set_at([0], 1.0)
    # memoryview, not np.asarray: numpy swallows a buffer-protocol error and falls back to a
    # 0-d object array, so the refusal is only observable through the raw protocol, where
    # pybind11 surfaces it as BufferError.
    with pytest.raises(BufferError):
        memoryview(t)


@pytest.mark.parametrize("device", GPU_DEVICES)
def test_linear_relu_forward_on_gpu_matches_cpu(device):
    weight = [1.0, -2.0, 3.0, 4.0, 0.5, -1.0]  # (in=2, out=3), row-major
    bias = [0.25, -0.5, 1.0]
    x_values = [1.0, 1.0, -1.0, 2.0]  # batch of 2

    def run(dev):
        linear = pulsatrix_py.LinearModule(2, 3, device=dev)
        linear.set_weight(weight)
        linear.set_bias(bias)
        relu = pulsatrix_py.ReluModule(device=dev)
        x = pulsatrix_py.Tensor.from_values([2, 2], x_values).to(dev)
        return relu.forward(linear.forward(x)).to(DeviceType.Cpu)

    np.testing.assert_allclose(np.asarray(run(device)), np.asarray(run(DeviceType.Cpu)), atol=1e-5)
