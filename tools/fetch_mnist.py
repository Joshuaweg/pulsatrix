"""Offline, one-time MNIST fetcher for the real MNIST training example
(plan_mnist_classification_training_example.md).

Not run by CTest/CI -- this is a data-acquisition tool, not a test dependency. Run once
manually (`py -3.11 tools/fetch_mnist.py`) before building the mnist_training_demo target
or running tests/mnist_loader_test.cpp / tests/mnist_classifier_example_test.cpp. Zero live
Python dependency at C++ build/test time afterward -- torchvision downloads and caches the
*original* MNIST IDX/ubyte files (the same format Yann LeCun's site has always served);
this script does no format conversion at all. The C++ side's MnistIdxLoader parses those
exact cached files directly.

data/ is gitignored -- real MNIST files are never committed to this repo.
"""

import pathlib

import torchvision

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
RAW_DIR = DATA_ROOT / "MNIST" / "raw"

EXPECTED_FILES = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
]


def main():
    print(f"Downloading MNIST (if not already cached) to {DATA_ROOT} ...")
    torchvision.datasets.MNIST(root=str(DATA_ROOT), train=True, download=True)
    torchvision.datasets.MNIST(root=str(DATA_ROOT), train=False, download=True)

    print("Verifying raw IDX files are present and decompressed:")
    missing = []
    for name in EXPECTED_FILES:
        path = RAW_DIR / name
        if path.exists():
            print(f"  OK   {path}  ({path.stat().st_size:,} bytes)")
        else:
            missing.append(path)
            print(f"  MISSING  {path}")

    if missing:
        raise SystemExit(
            f"{len(missing)} expected raw IDX file(s) missing after download -- "
            "torchvision's cache layout may have changed. See MnistIdxLoader's Recon."
        )

    print("\nAll 4 raw MNIST IDX files present. MnistIdxLoader can now read them directly.")


if __name__ == "__main__":
    main()
