"""真实 Dataset/训练/检查点/推理链；合成夹具仅写入 pytest 临时目录。"""

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

pytest.importorskip("geopandas")
pytest.importorskip("torchnet")

import train_semantic as train
from src import model_utils
from src.dataset import PASTIS_Dataset
from src.mm_utae_experiment import (
    initialize_mm_utae_heads,
    mm_utae_loss,
    validate_mm_utae_config,
)

ROOT = Path(__file__).resolve().parents[1]


def config(tmp_path, head="legacy", auxiliary=0.0):
    result = train.parser.parse_args([
        "--model", "mm_utae", "--device", "cpu", "--num_workers", "0",
        "--epochs", "1", "--fold", "2", "--batch_size", "2",
        "--accumulation_steps", "2", "--n_head", "4", "--d_model", "32",
        "--d_k", "2", "--encoder_widths", "[16,16,16,32]",
        "--decoder_widths", "[16,16,16,32]", "--out_conv", "[16,20]",
        "--mm-utae-head", head, "--mm-utae-aux-weight", str(auxiliary),
        "--ignore_index", "19", "--diagnostics",
        "--final-bn-init", "ones" if head == "legacy" else "legacy",
        "--res_dir", str(tmp_path / "results"),
    ])
    for name in train.list_args:
        setattr(result, name, json.loads(getattr(result, name)))
    return result


@pytest.fixture
def dataset_folder(tmp_path):
    folder = tmp_path / "fixture_PASTIS_R"
    folder.mkdir()
    rng = np.random.default_rng(31)
    modalities = {"S2": 10, "S1A": 3, "S1D": 3}
    for modality, channels in modalities.items():
        (folder / f"DATA_{modality}").mkdir()
        norms = {f"Fold_{fold}": {"mean": [0.0] * channels, "std": [1.0] * channels}
                 for fold in range(1, 6)}
        (folder / f"NORM_{modality}_patch.json").write_text(json.dumps(norms))
    (folder / "ANNOTATIONS").mkdir()
    features = []
    for fold in range(1, 6):
        properties: dict[str, int | str] = {"ID_PATCH": fold, "Fold": fold}
        for index, (modality, channels) in enumerate(modalities.items()):
            length = 2 + (fold + index) % 3
            images = rng.normal(size=(length, channels, 32, 32)).astype(np.float32)
            np.save(folder / f"DATA_{modality}" / f"{modality}_{fold}.npy", images)
            dates = {str(t): 20180901 + t * 3 for t in range(length)}
            properties[f"dates-{modality}"] = json.dumps(dates)
        labels = rng.integers(0, 20, size=(1, 32, 32), dtype=np.int64)
        np.save(folder / "ANNOTATIONS" / f"TARGET_{fold}.npy", labels)
        features.append({"type": "Feature", "geometry": None, "properties": properties})
    metadata = {"type": "FeatureCollection", "features": features}
    (folder / "metadata.geojson").write_text(json.dumps(metadata))
    return folder


def test_factory_metadata_and_rejected_options(tmp_path):
    cfg = config(tmp_path)
    train._set_mm_experiment_metadata(cfg)
    model = model_utils.get_model(cfg)
    assert model.active_modalities == ("S2", "S1A", "S1D")
    assert cfg.mamba_variant is None
    assert cfg.mm_utae_reconstruction["exact_author_implementation"] is False
    assert cfg.input_channels == {"S2": 10, "S1A": 3, "S1D": 3}
    for key, value in [("fusion_mode", "qaf"), ("modality_dropout", 0.2),
                       ("decoder_norm", "group"), ("mm_utae_aux_weight", float("nan"))]:
        invalid = config(tmp_path)
        setattr(invalid, key, value)
        with pytest.raises(ValueError):
            validate_mm_utae_config(invalid)
    with pytest.raises(ValueError, match="no final BatchNorm"):
        invalid = config(tmp_path, head="logits")
        invalid.final_bn_init = "ones"
        validate_mm_utae_config(invalid)


def test_auxiliary_loss_weights_and_fused_evaluation():
    target = torch.tensor([[[0, 1], [2, 19]]])
    weights = torch.ones(20)
    weights[19] = 0
    criterion = nn.CrossEntropyLoss(weight=weights)
    result = {key: torch.randn(1, 20, 2, 2, requires_grad=True)
              for key in ("logits", "logits_s2", "logits_s1a", "logits_s1d")}
    logits, loss = mm_utae_loss(result, target, criterion, 0.5)
    main = criterion(result["logits"], target)
    auxiliary = torch.stack([criterion(result[key], target) for key in result if key != "logits"])
    torch.testing.assert_close(loss, main + 0.5 * auxiliary.sum())
    assert logits is result["logits"]
    loss.backward()
    assert all(value.grad is not None and value.grad.abs().sum() > 0 for value in result.values())
    _, evaluation_loss = mm_utae_loss(result, target, criterion, 0)
    torch.testing.assert_close(evaluation_loss, main)
    with pytest.raises(ValueError, match="per-modality logits"):
        mm_utae_loss({"logits": logits}, target, criterion, 0.5)


def test_auxiliary_heads_do_not_override_fusion_diagnostics_or_initialization(tmp_path):
    model = model_utils.get_model(config(tmp_path, auxiliary=0.5))
    names = initialize_mm_utae_heads(model, 20)
    assert len(names) == 4
    for head in (model.fusion_head, *model.aux_heads.values()):
        _, batchnorm = train.find_final_batchnorm(head, 20)
        torch.testing.assert_close(batchnorm.weight, torch.ones(20))
        torch.testing.assert_close(batchnorm.bias, torch.zeros(20))
    name, _ = train._classifier_parameter(model, 20)
    bn_name, _ = train._classifier_batchnorm(model, 20)
    assert name.startswith("fusion_head.") and bn_name.startswith("fusion_head.")
    raw = model_utils.get_model(config(tmp_path, head="logits", auxiliary=0.5))
    assert train._classifier_batchnorm(raw, 20) == (None, None)


@pytest.mark.parametrize("head,auxiliary", [("legacy", 0.0), ("logits", 0.5)])
def test_complete_training_save_and_cli_inference(tmp_path, dataset_folder, monkeypatch, head, auxiliary):
    torch.set_num_threads(2)
    cfg = config(tmp_path, head, auxiliary)
    cfg.dataset_folder = str(dataset_folder)
    seen = []

    class RecordingDataset(PASTIS_Dataset):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            seen.append((kwargs["folds"], self.temporal_dropout))

    monkeypatch.setattr(train, "PASTIS_Dataset", RecordingDataset)
    train.main(cfg)
    assert seen == [([3, 4, 5], {"S2": 0.4, "S1A": 0.2, "S1D": 0.2}),
                    ([1], {}), ([2], {})]
    output = Path(cfg.res_dir)
    fold_dir = output / "Fold_2"
    assert all((fold_dir / name).is_file() for name in
               ("model.pth.tar", "trainlog.json", "test_metrics.json", "conf_mat.pkl"))
    saved = json.loads((output / "conf.json").read_text())
    assert saved["active_modalities"] == ["S2", "S1A", "S1D"]
    assert saved["mm_utae_head"] == head and saved["mm_utae_aux_weight"] == auxiliary
    assert saved["mamba_variant"] is None
    metrics = json.loads((fold_dir / "test_metrics.json").read_text())
    assert np.isfinite(metrics["test_loss"])
    assert metrics["test_classifier_name"].startswith("fusion_head.")
    with (fold_dir / "conf_mat.pkl").open("rb") as handle:
        matrix = pickle.load(handle)
    assert matrix.shape == (20, 20)
    inference = tmp_path / "inference"
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    process = subprocess.run([
        sys.executable, "test_semantic.py", "--dataset_folder", str(dataset_folder),
        "--weight_folder", str(output), "--res_dir", str(inference),
        "--fold", "2", "--device", "cpu", "--num_workers", "0",
    ], cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)
    assert process.returncode == 0, process.stdout + process.stderr
    inferred = json.loads((inference / "Fold_2" / "test_metrics.json").read_text())
    for key in ("test_accuracy", "test_IoU", "test_loss"):
        assert inferred[key] == pytest.approx(metrics[key], abs=1e-6)


def test_new_route_import_does_not_require_mamba():
    code = (
        "import sys; import train_semantic; "
        "assert not any('vmamba' in m or m == 'src.backbones.convlstm' for m in sys.modules); "
        "print('pure PyTorch import passed')"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("as_tensor", [False, True])
def test_result_saving_accepts_numpy_and_tensor(tmp_path, as_tensor):
    cfg = config(tmp_path)
    train.prepare_output(cfg)
    matrix = np.eye(20, dtype=np.int64)
    train.save_results(2, {"test_loss": 1.0}, torch.from_numpy(matrix) if as_tensor else matrix, cfg)
    with (Path(cfg.res_dir) / "Fold_2" / "conf_mat.pkl").open("rb") as handle:
        np.testing.assert_array_equal(pickle.load(handle), matrix)
