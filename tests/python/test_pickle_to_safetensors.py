"""Tests for tools/convert/pickle_to_safetensors.py, the legacy-pickle converter (IO-3).

Skipped where torch or safetensors isn't installed.
"""
import dataclasses
import os
import sys

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools", "convert"))

import pickle_to_safetensors as conv  # noqa: E402


def small_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.BatchNorm2d(4), torch.nn.ReLU(), torch.nn.Flatten(),
                               torch.nn.Linear(4 * 6 * 6, 5))


def convert(tmp_path, obj, *flags, name="in.pt"):
    src, dst = tmp_path / name, tmp_path / "out.safetensors"
    torch.save(obj, src)
    code = conv.main([str(src), str(dst), *flags])
    return code, dst


def same_bytes(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.reshape(-1).contiguous().view(torch.uint8),
                                                                     b.reshape(-1).contiguous().view(torch.uint8))


def test_a_state_dict_round_trips_bit_for_bit(tmp_path):
    model = small_model()
    model[1].running_mean.uniform_()          # non-trivial buffers, and a 0-dim int64 counter
    sd = model.state_dict()
    code, dst = convert(tmp_path, sd)
    assert code == 0
    back = safetensors_torch.load_file(str(dst))
    assert set(back) == set(sd)
    for k, v in sd.items():
        assert same_bytes(back[k], v), k
    assert back["1.num_batches_tracked"].shape == () and back["1.num_batches_tracked"].dtype == torch.int64
    from safetensors import safe_open
    with safe_open(str(dst), "pt") as f:
        meta = f.metadata()
    assert meta["format"] == "pt"
    assert meta["source_sha256"] == conv.sha256(tmp_path / "in.pt")
    assert meta["converted_from"] == "in.pt"


def test_a_training_checkpoint_uses_its_state_dict_and_skips_the_rest(tmp_path, capsys):
    model = small_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    model(torch.randn(2, 3, 8, 8)).sum().backward()
    opt.step()
    ckpt = {"epoch": 3, "state_dict": model.state_dict(), "optimizer": opt.state_dict(), "note": "run 7"}
    code, dst = convert(tmp_path, ckpt)
    assert code == 0
    back = safetensors_torch.load_file(str(dst))
    assert set(back) == set(model.state_dict())
    out = capsys.readouterr().out
    assert "from 'state_dict'" in out


def test_key_strip_prefix_and_float32(tmp_path):
    inner = {"module.weight": torch.randn(3, 2, dtype=torch.bfloat16), "module.bias": torch.arange(3)}
    code, dst = convert(tmp_path, {"model": {"ema": inner}, "other": {"x": torch.ones(1)}}, "--key", "model.ema",
                        "--strip-prefix", "module.", "--float32")
    assert code == 0
    back = safetensors_torch.load_file(str(dst))
    assert set(back) == {"weight", "bias"}
    assert back["weight"].dtype == torch.float32
    assert torch.equal(back["weight"], inner["module.weight"].float())
    assert back["bias"].dtype == torch.int64          # only floating-point tensors are cast


def test_bf16_is_kept_unless_asked(tmp_path):
    w = torch.randn(4, 4, dtype=torch.bfloat16)
    code, dst = convert(tmp_path, {"w": w})
    assert code == 0
    assert same_bytes(safetensors_torch.load_file(str(dst))["w"], w)


def test_views_and_shared_storage_are_written_as_their_own_tensors(tmp_path):
    base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    sd = {"emb": base, "head": base, "t": base.t(), "row": base[1], "slice": base[:, 2:4]}
    code, dst = convert(tmp_path, sd)
    assert code == 0
    back = safetensors_torch.load_file(str(dst))
    for k, v in sd.items():
        assert torch.equal(back[k], v), k
        assert back[k].shape == v.shape


def test_nested_lists_are_flattened_with_dots(tmp_path):
    code, dst = convert(tmp_path, {"layers": [{"w": torch.ones(2)}, {"w": torch.zeros(2)}]})
    assert code == 0
    assert set(safetensors_torch.load_file(str(dst))) == {"layers.0.w", "layers.1.w"}


@dataclasses.dataclass
class Payload:
    weights: object


def test_arbitrary_objects_are_refused(tmp_path, capsys):
    code, dst = convert(tmp_path, Payload(torch.ones(2)))
    assert code == 1
    assert not dst.exists()
    assert "weights_only" in capsys.readouterr().err


def test_torchscript_archives_are_refused(tmp_path, capsys):
    src, dst = tmp_path / "scripted.pt", tmp_path / "out.safetensors"
    torch.jit.save(torch.jit.script(torch.nn.Linear(2, 2)), str(src))
    assert conv.main([str(src), str(dst)]) == 1
    assert "TorchScript" in capsys.readouterr().err


def test_old_torch_is_refused(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(torch, "__version__", "2.5.1+cpu")
    code, _ = convert(tmp_path, {"w": torch.ones(1)})
    assert code == 1
    assert "CVE-2025-32434" in capsys.readouterr().err


def test_ambiguous_or_empty_checkpoints_are_refused(tmp_path, capsys):
    code, _ = convert(tmp_path, {"model": {"w": torch.ones(1)}, "ema": {"w": torch.zeros(1)}})
    assert code == 1
    assert "--key" in capsys.readouterr().err
    code, _ = convert(tmp_path, {"epoch": 1}, name="empty.pt")
    assert code == 1
    assert "no tensors" in capsys.readouterr().err


def test_sparse_tensors_are_refused(tmp_path, capsys):
    code, _ = convert(tmp_path, {"s": torch.eye(3).to_sparse()})
    assert code == 1
    assert "sparse" in capsys.readouterr().err


def test_outputs_are_never_overwritten_by_accident(tmp_path, capsys):
    src = tmp_path / "in.pt"
    torch.save({"w": torch.ones(1)}, src)
    assert conv.main([str(src), str(src)]) == 1
    dst = tmp_path / "out.safetensors"
    dst.write_bytes(b"keep me")
    assert conv.main([str(src), str(dst)]) == 1
    assert dst.read_bytes() == b"keep me"
    assert conv.main([str(src), str(dst), "--force"]) == 0


def test_the_legacy_non_zip_format_converts(tmp_path):
    src, dst = tmp_path / "old.pth", tmp_path / "out.safetensors"
    w = torch.randn(3, 3)
    torch.save({"w": w}, src, _use_new_zipfile_serialization=False)
    assert conv.main([str(src), str(dst)]) == 0
    assert torch.equal(safetensors_torch.load_file(str(dst))["w"], w)
