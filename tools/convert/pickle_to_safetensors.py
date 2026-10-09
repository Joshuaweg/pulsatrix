"""Converts a PyTorch pickle checkpoint (.pt, .pth, .pkl) to safetensors (roadmap IO-3):

    python3 tools/convert/pickle_to_safetensors.py resnet18.pth resnet18.safetensors

pulsatrix never unpickles in C++: a pickle file is a program, and loading one can run code. This
converter is the only place pickle is read, and it reads it with PyTorch's restricted loader,
`torch.load(weights_only=True)`, which only rebuilds tensors and plain containers.

- **torch 2.6 or newer is required.** Before 2.6, `weights_only=True` could still run code
  (CVE-2025-32434), so older versions are refused.
- **What it refuses:** TorchScript archives (export the model's `state_dict()` instead), files
  that need anything beyond tensors, dicts, lists, numbers and strings, sparse, quantized and
  complex tensors, and an output path that exists (unless --force). There is no option to load
  unsafely: if a file needs more than weights_only allows, re-save its `state_dict()` with
  `torch.save` in an environment you trust.
- **Which tensors:** the file's top level, or with --key the dict at that path (dots separate
  levels, e.g. `state_dict` or `model.ema`). Without --key, a top level holding no tensors and a
  single dict of tensors under `state_dict`, `model`, `model_state_dict`, `module`, `net` or
  `ema` uses that dict. Nested dicts and lists are flattened with dots. Other values (epoch
  numbers, optimizer settings, strings) are skipped and listed.
- **Names:** --strip-prefix removes a prefix such as `module.` (DataParallel) where present.
- **Tensors** are written contiguous, each with its own storage (safetensors can't share), in
  their own dtype; --float32 casts floating-point tensors to float32. pulsatrix reads float32,
  bf16 and fp16 (upcast) and integer tensors.
- **Metadata:** `format: pt`, the source file's name and SHA-256, the torch version.
- **Verification:** the written file is read back and compared bit for bit (--no-verify skips).

Needs torch (2.6+) and safetensors. Exit status: 0 on success, 1 on a refused or failed
conversion, 2 on bad arguments.
"""

import argparse
import hashlib
import os
import sys
import zipfile

PREFERRED_KEYS = ("state_dict", "model", "model_state_dict", "module", "net", "ema")


class Refused(Exception):
    """A file this converter won't convert, with the reason."""


def require_torch():
    try:
        import torch
    except ImportError as e:
        raise Refused("needs PyTorch 2.6 or newer: pip install torch") from e
    version = torch.__version__.split("+")[0]
    parts = []
    for p in version.split(".")[:2]:
        digits = "".join(c for c in p if c.isdigit())
        parts.append(int(digits) if digits else 0)
    if tuple(parts) < (2, 6):
        raise Refused(f"torch {torch.__version__} is too old: before 2.6, torch.load(weights_only=True) "
                      "could run code (CVE-2025-32434). Upgrade to torch 2.6 or newer.")
    return torch


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def refuse_torchscript(path):
    if not zipfile.is_zipfile(path):
        return
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
    if any("/code/" in n or n.startswith("code/") or n.endswith("/constants.pkl") or n == "constants.pkl" for n in names):
        raise Refused("this is a TorchScript archive (torch.jit.save), not a state dict. Load it where you trust it "
                      "and save its weights instead: torch.save(torch.jit.load(path).state_dict(), 'weights.pt')")


def load(torch, path):
    refuse_torchscript(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:  # weights_only raises UnpicklingError, RuntimeError or ValueError
        raise Refused(f"torch.load(weights_only=True) refused the file: {e}\n"
                      "It needs more than tensors and plain containers to load. If you trust it, load it in a "
                      "separate environment and re-save model.state_dict() with torch.save.") from None


def select(torch, obj, key):
    if key:
        for part in key.split("."):
            if isinstance(obj, dict) and part in obj:
                obj = obj[part]
            elif isinstance(obj, (list, tuple)) and part.isdigit() and int(part) < len(obj):
                obj = obj[int(part)]
            else:
                raise Refused(f"--key {key}: no entry {part!r}")
        return obj, key
    if isinstance(obj, dict) and not any(isinstance(v, torch.Tensor) for v in obj.values()):
        found = [k for k in PREFERRED_KEYS
                 if isinstance(obj.get(k), dict) and any(isinstance(v, torch.Tensor) for v in obj[k].values())]
        if len(found) == 1:
            return obj[found[0]], found[0]
        if len(found) > 1:
            raise Refused(f"several dicts of tensors ({', '.join(found)}); choose one with --key")
    return obj, ""


def flatten(torch, obj, prefix, tensors, skipped):
    if isinstance(obj, torch.Tensor):
        tensors[prefix] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            flatten(torch, v, f"{prefix}.{k}" if prefix else str(k), tensors, skipped)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            flatten(torch, v, f"{prefix}.{i}" if prefix else str(i), tensors, skipped)
    else:
        skipped.append(prefix or "(top level)")


def prepare(torch, tensors, strip_prefix, float32):
    out = {}
    seen_storage = set()
    for name, t in tensors.items():
        if t.is_sparse or t.layout != torch.strided:
            raise Refused(f"{name}: sparse tensors can't be written to safetensors; densify it first")
        if t.is_quantized:
            raise Refused(f"{name}: quantized tensors aren't supported; dequantize it first")
        if t.is_complex():
            raise Refused(f"{name}: complex tensors aren't supported by safetensors")
        new_name = name[len(strip_prefix):] if strip_prefix and name.startswith(strip_prefix) else name
        if new_name in out:
            raise Refused(f"{name}: two tensors map to {new_name!r} after --strip-prefix")
        t = t.detach().cpu()
        if float32 and t.is_floating_point():
            t = t.float()
        key = t.untyped_storage().data_ptr()
        if not t.is_contiguous() or key in seen_storage or t.untyped_storage().nbytes() != t.numel() * t.element_size():
            t = t.contiguous().clone()  # its own storage, exactly its own bytes
        seen_storage.add(t.untyped_storage().data_ptr())
        out[new_name] = t
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert a PyTorch pickle checkpoint to safetensors, safely.")
    parser.add_argument("input", help=".pt, .pth or .pkl file written by torch.save")
    parser.add_argument("output", help="the .safetensors file to write")
    parser.add_argument("--key", default="", help="convert the dict at this path, e.g. state_dict or model.ema")
    parser.add_argument("--strip-prefix", default="", help="remove this prefix from names, e.g. module.")
    parser.add_argument("--float32", action="store_true", help="cast floating-point tensors to float32")
    parser.add_argument("--force", action="store_true", help="overwrite the output if it exists")
    parser.add_argument("--no-verify", action="store_true", help="don't read the output back to check it")
    args = parser.parse_args(argv)

    try:
        if not os.path.isfile(args.input):
            raise Refused(f"{args.input}: no such file")
        if os.path.abspath(args.input) == os.path.abspath(args.output):
            raise Refused("the output would overwrite the input")
        if os.path.exists(args.output) and not args.force:
            raise Refused(f"{args.output} exists; pass --force to overwrite it")
        torch = require_torch()
        from safetensors.torch import load_file, save_file

        obj = load(torch, args.input)
        obj, used_key = select(torch, obj, args.key)
        tensors, skipped = {}, []
        flatten(torch, obj, "", tensors, skipped)
        if not tensors:
            raise Refused("no tensors found" + (f" under {used_key!r}" if used_key else "") +
                          "; use --key to pick the dict that holds them")
        tensors = prepare(torch, tensors, args.strip_prefix, args.float32)
        metadata = {"format": "pt", "converted_from": os.path.basename(args.input), "source_sha256": sha256(args.input),
                    "torch": torch.__version__, "converter": "pulsatrix tools/convert/pickle_to_safetensors.py"}
        if used_key:
            metadata["key"] = used_key
        save_file(tensors, args.output, metadata=metadata)
        if not args.no_verify:
            back = load_file(args.output)

            def raw(t):  # the tensor's bytes, so NaNs and -0.0 compare exactly
                return t.reshape(-1).contiguous().view(torch.uint8)

            bad = [n for n, t in tensors.items() if n not in back or back[n].dtype != t.dtype or back[n].shape != t.shape
                   or not torch.equal(raw(back[n]), raw(t))]
            if bad or len(back) != len(tensors):
                raise Refused(f"verification failed for {bad[:5] or 'the tensor count'}")
    except Refused as e:
        print(f"pickle_to_safetensors: {e}", file=sys.stderr)
        return 1

    total = sum(t.numel() for t in tensors.values())
    dtypes = sorted({str(t.dtype).replace("torch.", "") for t in tensors.values()})
    where = f" (from {used_key!r})" if used_key else ""
    print(f"wrote {args.output}: {len(tensors)} tensors{where}, {total:,} values, dtypes {', '.join(dtypes)}"
          + ("" if args.no_verify else ", verified"))
    if skipped:
        shown = ", ".join(skipped[:8]) + (f" and {len(skipped) - 8} more" if len(skipped) > 8 else "")
        print(f"skipped {len(skipped)} non-tensor entries: {shown}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
