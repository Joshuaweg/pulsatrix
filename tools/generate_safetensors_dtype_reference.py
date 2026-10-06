"""Offline, one-time reference generator for tests/safetensors_test.cpp's dtype conversions
(IO-6): a safetensors file written by the reference implementation holding BF16, F16, F64 and
both FP8 formats, with each tensor's values as torch converts them to float32.

Not run by CTest/CI. Run with torch and safetensors installed:

    python3 tools/generate_safetensors_dtype_reference.py

Versions the committed values were generated with: Python 3.12, torch 2.14.1, safetensors 0.8.0.
Values cover the edge cases: signs, the largest finite value, subnormals, infinities and NaN.
"""

import math

import torch
from safetensors.torch import save

TENSORS = {
    "bf16": torch.tensor([1.0, -2.5, 3.0e38, 9.1835e-41, float("inf"), -float("inf"), float("nan"), -0.0],
                         dtype=torch.bfloat16),
    "f16": torch.tensor([1.0, -2.5, 65504.0, 5.960464477539063e-08, 6.1e-05, float("inf"), float("nan"), -0.0],
                        dtype=torch.float16),
    "f64": torch.tensor([1.0, -2.5, 0.1, 1e300, -1e-300], dtype=torch.float64),
    "f8e4m3": torch.tensor([1.0, -2.5, 448.0, 0.001953125, -0.0, float("nan")]).to(torch.float8_e4m3fn),
    "f8e5m2": torch.tensor([1.0, -2.5, 57344.0, 1.52587890625e-05, float("inf"), float("nan")]).to(torch.float8_e5m2),
}


def cpp_float(v):
    if math.isnan(v):
        return "kNaN"
    if math.isinf(v):
        return "kInf" if v > 0 else "-kInf"
    if v == 0 and math.copysign(1, v) < 0:
        return "-0.0f"
    text = "%.9g" % v
    return (text if any(c in text for c in ".e") else text + ".0") + "f"


if __name__ == "__main__":
    data = save(TENSORS)
    print("const std::vector<uint8_t> kDtypes = {")
    for i in range(0, len(data), 16):
        print("    " + ", ".join("0x%02x" % b for b in data[i:i + 16]) + ",")
    print("};")
    for name, t in TENSORS.items():
        values = t.float().tolist()
        print("const std::vector<float> k%sValues = {%s};" % (name.upper(), ", ".join(cpp_float(v) for v in values)))
