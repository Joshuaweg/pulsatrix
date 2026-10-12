"""Offline, one-time generator for tests/fixtures/npy/ (INT-8): .npy files written by numpy itself,
which tests/embedding_export_test.cpp reads with ParseNpy() and compares SerializeNpy() against.

Not run by CTest/CI. Run it with numpy installed, from the repository root:

    python3 tools/generate_npy_fixtures.py tests/fixtures/npy

Versions the committed files were generated with: Python 3.12, numpy 2.5.2.

Every array is made from a formula, so the C++ test knows the values: v[k] = 0.25 k - 1 for the
k-th value in C order, which float32 holds exactly.
"""

import os
import sys

import numpy as np
from numpy.lib import format as npformat


def values(shape, dtype):
    count = int(np.prod(shape, dtype=np.int64))
    return (0.25 * np.arange(count) - 1.0).astype(dtype).reshape(shape)


def main(out):
    os.makedirs(out, exist_ok=True)
    # What SerializeNpy must match byte for byte.
    np.save(os.path.join(out, "c_f4_2x3.npy"), values((2, 3), np.float32))
    np.save(os.path.join(out, "c_f8_3x2.npy"), values((3, 2), np.float64))
    np.save(os.path.join(out, "fortran_f4_2x3x4.npy"), np.asfortranarray(values((2, 3, 4), np.float32)))
    np.save(os.path.join(out, "vector_f4_5.npy"), values((5,), np.float32))
    np.save(os.path.join(out, "scalar_f4.npy"), np.float32(-1.0))
    np.save(os.path.join(out, "empty_f4_0x3.npy"), values((0, 3), np.float32))
    np.save(os.path.join(out, "c_i8_2x2.npy"), np.arange(4, dtype=np.int64).reshape(2, 2))
    np.save(os.path.join(out, "c_bigendian_f4_2.npy"), values((2,), ">f4"))
    with open(os.path.join(out, "v2_f4_2x3.npy"), "wb") as f:
        npformat.write_array(f, values((2, 3), np.float32), version=(2, 0))
    for name in sorted(os.listdir(out)):
        print(name, os.path.getsize(os.path.join(out, name)))


if __name__ == "__main__":
    main(sys.argv[1])
