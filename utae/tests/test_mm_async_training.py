"""临时合成PASTIS夹具的训练保存推理测试；真实Mamba由CUDA脚本验证。"""

import importlib
import json
import os
from pathlib import Path
import subprocess

import pytest
import torch

pytest.importorskip("geopandas")
pytest.importorskip("torchnet")

from test_mm_utae_io import dataset_folder as _dataset_fixture
from test_mm_ucpmamba_mid import mid_module as _mid_fixture

dataset_folder, mid_module = _dataset_fixture, _mid_fixture


def test_independent_profile_and_one_epoch_saves_best_last_and_full_metrics(
        dataset_folder, tmp_path, monkeypatch, mid_module):
    import train_semantic as train
    from run_mm_ucpmamba_async import make_config
    from test_mm_ucpmamba_async import instance

    old_default = train.parser.get_default("decoder_norm")
    cfg = make_config(["--device", "cpu", "--epochs", "1", "--batch_size", "1", "--fold", "2",
                       "--dataset_folder", str(dataset_folder), "--res_dir", str(tmp_path / "result"),
                       "--mid-num-bins", "4", "--mid-start-day", "0", "--mid-end-day", "40"])
    assert train.parser.get_default("decoder_norm") == old_default
    assert cfg.decoder_norm == "group" and cfg.final_bn_init == "ones"
    assert cfg.temporal_dropout_s2 == .4 and cfg.temporal_dropout_s1a == .2
    monkeypatch.setattr(train.model_utils, "get_model", lambda *a, **kw: instance(input_size=32, out_conv=[8, 20]))
    train.main(cfg)
    result = Path(cfg.res_dir) / "Fold_2"
    best = torch.load(result / "model.pth.tar", weights_only=False)
    last = torch.load(result / "last.pth.tar", weights_only=False)
    assert best["epoch"] == last["epoch"] == 1
    assert best["model_metadata"]["async_design"] == cfg.async_design
    assert last["config"]["mid_num_bins"] == 4
    assert (result / "config.json").exists()
    report = json.loads((result / "class_metrics.json").read_text())
    assert len(report["confusion_matrix"]) == 20 and len(report["per_class"]) == 20
    log = json.loads((result / "trainlog.json").read_text())["1"]
    assert "train_async" in log and "val_async" in log
    assert log["train_learning_rate"] == .001
    infer = importlib.import_module("test_semantic")
    monkeypatch.setattr(infer.model_utils, "get_model", lambda *a, **kw: instance(input_size=32, out_conv=[8, 20]))
    cfg.weight_folder = cfg.res_dir
    cfg.res_dir = str(tmp_path / "inference")
    infer.main(cfg)
    assert (Path(cfg.res_dir) / "Fold_2" / "class_metrics.json").exists()


@pytest.mark.skipif(os.name != "posix", reason="bash launcher test runs in WSL")
def test_launcher_uses_five_fold_protocol_and_rejects_overwrite(tmp_path):
    root = Path(__file__).resolve().parents[1]
    data, result = tmp_path / "data", tmp_path / "result"
    data.mkdir()
    calls = tmp_path / "calls.txt"
    executable = tmp_path / "python_fixture"
    executable.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$CALLS"\n')
    executable.chmod(0o755)
    environment = dict(os.environ, DATASET_FOLDER=str(data), RES_DIR=str(result),
                       FOLDS="1 2 3 4 5", PYTHON=str(executable), CALLS=str(calls))
    process = subprocess.run(["bash", str(root / "run_mm_ucpmamba_async.sh")],
                             env=environment, text=True, capture_output=True)
    assert process.returncode == 0, process.stderr
    lines = calls.read_text().splitlines()
    assert len(lines) == 6
    assert all(f"--fold {fold} --rdm_seed 1" in lines[fold - 1] for fold in range(1, 6))
    assert "-m src.mm_async_results" in lines[-1]
    (result / "conf.json").write_text("{}")
    process = subprocess.run(["bash", str(root / "run_mm_ucpmamba_async.sh")],
                             env=environment, text=True, capture_output=True)
    assert process.returncode != 0
    assert len(calls.read_text().splitlines()) == 6
