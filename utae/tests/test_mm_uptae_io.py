"""mm-uptae 的训练/推理别名、完整1016配置和检查点隔离测试。"""

import importlib
import sys
import types

import pytest
import torch

from test_mm_mid_io import config, inference_parser, sample, train_helpers, training_parser
from test_mm_uptae import (
    mid_module as _mid_fixture,
    real_convlstm as _recurrence_fixture,
    uptae_module as _uptae_fixture,
)


ALIASES = ("mm-uptae", "mm_uptae")
mid_module = _mid_fixture
real_convlstm = _recurrence_fixture
uptae_module = _uptae_fixture


@pytest.mark.parametrize("name", ALIASES)
def test_train_and_inference_accept_both_model_spellings(name):
    inference, _ = inference_parser()
    assert training_parser().parse_args(["--model", name]).model == name
    assert inference.parse_args(["--model", name]).model == name


@pytest.mark.parametrize("name", ALIASES)
def test_model_factory_keeps_full_1016_and_selects_ltae(uptae_module, monkeypatch, name):
    module, _, _ = uptae_module
    model_utils = importlib.import_module("src.model_utils")
    received = {}

    def constructor(**kwargs):
        received.update(kwargs)
        return "full uptae"

    monkeypatch.setattr(module, "MMUPTAE", constructor)
    cfg = config(name)
    cfg.__dict__.update(encoder_widths=[8, 8, 8, 16], decoder_widths=[8, 8, 8, 16],
                        out_conv=[8, 4], str_conv_k=4, str_conv_s=2, str_conv_p=1,
                        encoder_norm="group", decoder_norm="group", decoder_groups=8,
                        n_head=4, d_model=32, d_k=4, padding_mode="reflect", pad_value=0)
    assert model_utils.get_model(cfg) == "full uptae"
    assert received["hidden_dim"] == 64
    assert (received["n_head"], received["d_model"], received["d_k"]) == (4, 32, 4)
    assert received["dynamic_enabled"] and received["interaction_enabled"]
    assert received["decoder_norm"] == "group"
    assert (received["num_bins"], received["start_day"], received["end_day"]) == (24, 0, 40)


@pytest.mark.parametrize("name", ALIASES)
def test_forward_keeps_three_independent_date_axes_and_masks(name):
    received = {}

    def recorder(*args, **kwargs):
        received.update(kwargs)
        return args

    data, dates = sample((3, 2, 1))[0]
    output = train_helpers()["_forward_model"](recorder, data, dates, config(name))
    assert all(value is data[satellite] for value, satellite in zip(output, ("S2", "S1A", "S1D")))
    for satellite, argument in (("S2", "batch_positions_s2"), ("S1A", "batch_positions_s1a"),
                                ("S1D", "batch_positions_s1d")):
        assert received[argument] is dates[satellite]
    assert received["pad_masks"] is data["pad_masks"]
    assert received["valid_masks"] is data["valid_masks"]


@pytest.mark.parametrize("name", ALIASES)
def test_metadata_preserves_full_1016_and_records_ltae_contract(name):
    metadata = train_helpers()["_set_mm_experiment_metadata"]
    baseline, new = config("mm_ucpmamba_async"), config(name)
    metadata(baseline)
    metadata(new)
    assert new.res_dir != baseline.res_dir
    assert new.experiment_name == "mm_uptae_s2_s1a_s1d"
    assert new.active_modalities == baseline.active_modalities == ["S2", "S1A", "S1D"]
    assert new.mid_alignment == baseline.mid_alignment
    assert new.mamba_variant is None and baseline.mamba_variant is not None
    assert new.async_design == dict(baseline.async_design, temporal_encoder="ltae",
                                    ltae_settings=dict(n_head=16, d_model=256, d_k=4,
                                                       position="window_centers", output_dim=64,
                                                       mask="pixel_valid"))


def test_last_checkpoint_accepts_alias_and_rejects_other_models_heads_and_calendar(tmp_path):
    from src.mm_async_experiment import check_async_checkpoint, configure_async, save_async_last

    cfg = config(ALIASES[0])
    configure_async(cfg)
    cfg.res_dir = str(tmp_path)
    (tmp_path / "Fold_2").mkdir()
    model = torch.nn.Conv2d(1, 1, 1)
    optimizer = torch.optim.Adam(model.parameters())
    save_async_last(model, optimizer, cfg, fold=2, epoch=3, best_miou=.6)
    saved = torch.load(tmp_path / "Fold_2/last.pth.tar", weights_only=True)
    assert saved["model_metadata"]["model_name"] == ALIASES[0]
    assert saved["model_metadata"]["async_design"]["temporal_encoder"] == "ltae"
    assert saved["epoch"] == 3 and saved["best_val_miou"] == .6
    check_async_checkpoint(saved, cfg)
    check_async_checkpoint(saved, config(ALIASES[1]))
    for name in ("mm_ucpmamba_async", "mm-ucpmamba-novmamba"):
        with pytest.raises(ValueError, match="configuration/reference date mismatch"):
            check_async_checkpoint(saved, config(name))
    for key, value in (("n_head", 8), ("d_model", 128), ("d_k", 8),
                       ("ref_date", "2019-09-01")):
        changed = config(ALIASES[1])
        setattr(changed, key, value)
        with pytest.raises(ValueError, match="configuration/reference date mismatch"):
            check_async_checkpoint(saved, changed)


def test_independent_entrypoint_changes_only_model_default(monkeypatch):
    parser = training_parser()
    train = types.ModuleType("train_semantic")
    train.parser, train.list_args, train.main = parser, ["encoder_widths", "decoder_widths", "out_conv"], None
    monkeypatch.setitem(sys.modules, "train_semantic", train)
    for name in ("run_mm_ucpmamba_async", "run_mm_uptae"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    try:
        original = importlib.import_module("run_mm_ucpmamba_async")
        entrypoint = importlib.import_module("run_mm_uptae")
        before = parser.get_default("model")
        old, new = original.make_config([]), entrypoint.make_config([])
        assert parser.get_default("model") == before
        assert old.model == "mm_ucpmamba_async" and new.model == ALIASES[0]
        assert {key for key in vars(old) if getattr(old, key) != getattr(new, key)} == {"model"}
        assert new.encoder_widths == [64, 64, 64, 128]
        assert new.decoder_norm == "group" and new.mid_low_memory
        assert (new.mid_num_bins, new.mid_start_day, new.mid_end_day) == (24, -200., 600.)
        assert new.ignore_index == 19 and new.epochs == 100
        assert entrypoint.make_config(["--model", ALIASES[1]]).model == ALIASES[1]
        with pytest.raises(ValueError):
            entrypoint.make_config(["--model", "mm_ucpmamba_async"])
    finally:
        for name in ("run_mm_ucpmamba_async", "run_mm_uptae"):
            sys.modules.pop(name, None)
