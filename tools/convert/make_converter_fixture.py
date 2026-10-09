"""Writes tests/fixtures/convert/converted.safetensors for tests/safetensors_converter_test.cpp
(IO-3): a checkpoint with known values saved by torch.save, then converted by
pickle_to_safetensors.py, so the C++ test reads exactly what the converter writes.

    python3 tools/convert/make_converter_fixture.py tests/fixtures/convert

The checkpoint is a training-style dict: {"epoch": 4, "state_dict": {...}}, with
- `module.linear.weight`: arange(6) reshaped (2, 3) and transposed, so (3, 2) and not contiguous;
- `module.linear.bias`: [0.5, -1.5] in bfloat16;
- `module.bn.num_batches_tracked`: a 0-dim int64 7;
- `module.embed.weight` and `module.head.weight`: one shared (2, 2) tensor [[1, 2], [3, 4]].
Converted with --strip-prefix module. Needs torch (2.6+) and safetensors.
"""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(__file__))
import pickle_to_safetensors as conv  # noqa: E402


def main():
    out_dir = sys.argv[1]
    shared = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    sd = {
        "module.linear.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3).t(),
        "module.linear.bias": torch.tensor([0.5, -1.5], dtype=torch.bfloat16),
        "module.bn.num_batches_tracked": torch.tensor(7, dtype=torch.int64),
        "module.embed.weight": shared,
        "module.head.weight": shared,
    }
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "checkpoint.pt")
        torch.save({"epoch": 4, "state_dict": sd}, src)
        code = conv.main([src, os.path.join(out_dir, "converted.safetensors"), "--strip-prefix", "module.", "--force"])
    sys.exit(code)


if __name__ == "__main__":
    main()
