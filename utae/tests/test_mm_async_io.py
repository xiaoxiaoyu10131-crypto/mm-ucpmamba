"""Async 的排序、CLI、元数据、指标与汇总协议测试。"""

import importlib
import json

import numpy as np
import pytest
import torch

from src.mm_async_experiment import add_async_arguments, check_async_checkpoint, configure_async
from src.mm_async_results import metric_report, summarize, summarize_seeds
from test_mm_mid_io import config, inference_parser, sample, train_helpers
from test_mm_ucpmamba_mid import mid_module as _mid_fixture

mid_module = _mid_fixture


def test_collate_sorts_images_dates_and_both_masks_without_mutating_input(mid_module):
    collate = importlib.import_module("src.mm_async_collate").async_pad_collate
    item = sample((3, 2, 1))
    images, dates = item[0]
    dates["S2"] = torch.tensor([20., 0., 10.])
    images["S2"][:, 0, 0, 0] = torch.tensor([2., 0., 1.])
    images["pad_masks"]["S2"][0] = True
    images["valid_masks"]["S2"][2] = False
    (output, times), _ = collate([item])
    assert times["S2"].tolist() == [[0., 10., 20.]]
    assert output["S2"][0, :, 0, 0, 0].tolist() == [0., 1., 2.]
    assert output["pad_masks"]["S2"].tolist() == [[False, False, True]]
    assert not output["valid_masks"]["S2"][0, 1:].any()
    assert dates["S2"].tolist() == [20., 0., 10.]


def test_training_forward_preserves_independent_axes_and_explicit_masks():
    forward = train_helpers()["_forward_model"]
    received = {}

    def recorder(*args, **kwargs):
        received.update(kwargs)
        return args

    data, dates = sample((3, 2, 1))[0]
    output = forward(recorder, data, dates, config("mm_ucpmamba_async"))
    assert [len(x) for x in output] == [3, 2, 1]
    assert received["pad_masks"] is data["pad_masks"]
    assert received["valid_masks"] is data["valid_masks"]
    assert received["batch_positions_s1d"] is dates["S1D"]


def test_configuration_and_inference_do_not_replace_saved_async_settings():
    parser, _ = inference_parser()
    add_async_arguments(parser, inference=True)
    default = vars(parser.parse_args([]))
    assert all(v is None for k, v in default.items() if k.startswith("async_"))
    explicit = parser.parse_args(["--async-radius", "2", "--no-async-dynamic"])
    assert explicit.async_radius == 2 and explicit.async_dynamic is False
    cfg = config("mm_ucpmamba_async")
    cfg.mid_start_day, cfg.mid_end_day = None, None
    configure_async(cfg)
    assert (cfg.mid_start_day, cfg.mid_end_day) == (-200, 600)
    check_async_checkpoint({"model_metadata": {"async_design": cfg.async_design}}, cfg)
    cfg.ref_date = "2019-09-01"
    with pytest.raises(ValueError, match="reference date"):
        check_async_checkpoint({"model_metadata": {"async_design": {}}}, cfg)


def test_full_confusion_matrix_is_preserved_when_void_is_ignored():
    from src.learning.miou import IoU

    cm = np.array([[4, 1, 2], [1, 3, 1], [5, 2, 0]])
    meter = IoU(3, ignore_index=2)
    meter.conf_metric.conf = cm.copy()
    meter.get_miou_acc()
    meter.value()
    np.testing.assert_array_equal(meter.conf_metric.value(), cm)
    result = metric_report(cm, 2)
    assert result["confusion_matrix"] == cm.tolist()
    assert result["overall"]["OA"] == 7 / 9
    assert result["per_class"]["0"]["pixels"] == 7
    assert result["per_class"]["2"]["ignored"]
    assert result["per_class"]["2"]["IoU"] is None
    json.dumps(result, allow_nan=False)


def test_empty_class_metrics_are_json_safe_and_per_class_fields_complete():
    report = metric_report(np.zeros((3, 3)), 2)
    assert report["overall"]["mIoU"] is None
    assert set(report["per_class"]["0"]) == {"IoU", "Precision", "Recall", "F1", "ignored", "pixels", "predictions"}
    json.dumps(report, allow_nan=False)


def test_five_fold_equal_mean_and_merged_metrics_are_separate(tmp_path):
    for fold in range(1, 6):
        folder = tmp_path / f"Fold_{fold}"
        folder.mkdir()
        matrix = np.array([[fold, 1], [1, 10]])
        (folder / "class_metrics.json").write_text(json.dumps(metric_report(matrix, None)))
        (folder / "config.json").write_text(json.dumps({"fold": fold, "ignore_index": None}))
    result = summarize(tmp_path)
    assert result["fold_equal_mean"]["mIoU"] != result["merged_confusion_metrics"]["overall"]["mIoU"]
    path = tmp_path / "Fold_3" / "config.json"
    path.write_text(json.dumps({"fold": 3, "ignore_index": None, "async_radius": 2}))
    with pytest.raises(ValueError, match="different configurations"):
        summarize(tmp_path)


def test_seed_summary_records_dispersion_and_rejects_mixed_protocols(tmp_path):
    roots = [tmp_path / f"seed_{seed}" for seed in (1, 2)]
    for seed, root in enumerate(roots, 1):
        for fold in range(1, 6):
            path = root / f"Fold_{fold}"
            path.mkdir(parents=True)
            (path / "class_metrics.json").write_text(json.dumps(metric_report([[seed, 1], [1, 5]], None)))
            (path / "config.json").write_text(json.dumps({"rdm_seed": seed, "fold": fold, "ignore_index": None}))
    report = summarize_seeds(roots)
    assert report["seeds"] == [1, 2] and report["fold_equal_mean"]["mIoU"]["std"] > 0
    assert (tmp_path / "async_multiseed_summary.json").exists()
    (roots[1] / "Fold_1" / "config.json").write_text(json.dumps({"rdm_seed": 2, "async_radius": 3}))
    with pytest.raises(ValueError, match="different protocols"):
        summarize_seeds(roots)
