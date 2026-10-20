"""源码记录不得因服务器缺少辅助启动脚本而阻断训练。"""

import hashlib
import json
from pathlib import Path

import pytest

from src import mm_async_experiment as experiment


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    root = tmp_path / "utae"
    source = root / "src"
    source.mkdir(parents=True)
    (source / "model.py").write_bytes(b"model source\n")
    (root / "train_semantic.py").write_bytes(b"training source\n")
    monkeypatch.setattr(experiment, "__file__", str(source / "mm_async_experiment.py"))
    monkeypatch.setattr(experiment.subprocess, "check_output", lambda *a, **kw: "")
    return root


def test_missing_launchers_are_recorded_and_existing_source_is_hashed(source_tree):
    report = experiment.source_provenance()
    assert "mm-ucpmamba-async.sh" in report["missing_files"]
    assert "run_mm_ucpmamba_async.sh" in report["missing_files"]
    assert "run_mm_ucpmamba_async.py" in report["missing_files"]
    assert report["source_sha256"]["train_semantic.py"] == hashlib.sha256(b"training source\n").hexdigest()
    key = str(Path("src") / "model.py")
    assert report["source_sha256"][key] == hashlib.sha256(b"model source\n").hexdigest()
    assert report["unreadable_files"] == {}
    json.dumps(report)


def test_all_launchers_present_are_hashed_without_missing_entries(source_tree):
    for name in ("test_semantic.py", "run_mm_ucpmamba_async.py",
                 "run_mm_ucpmamba_async.sh", "mm-ucpmamba-async.sh"):
        (source_tree / name).write_bytes(b"launcher\n")
    report = experiment.source_provenance()
    assert report["missing_files"] == []
    assert len(report["source_sha256"]) == 6
    assert report["source_sha256"]["mm-ucpmamba-async.sh"] == hashlib.sha256(b"launcher\n").hexdigest()


def test_unreadable_auxiliary_file_is_recorded(source_tree, monkeypatch):
    original = Path.read_bytes

    def read(path):
        if path.name == "mm-ucpmamba-async.sh":
            raise PermissionError("fixture denies reading the launcher")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    report = experiment.source_provenance()
    assert report["unreadable_files"]["mm-ucpmamba-async.sh"].startswith("PermissionError:")
    assert "train_semantic.py" in report["source_sha256"]


def test_server_without_git_still_records_source_hashes(source_tree, monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(experiment.subprocess, "check_output", unavailable)
    report = experiment.source_provenance()
    assert report["git_revision"] is None and report["dirty"] is None
    assert "train_semantic.py" in report["source_sha256"]
