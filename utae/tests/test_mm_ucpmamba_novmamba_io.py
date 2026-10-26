"""无 VMamba 1016 模型的别名、训练接线与 checkpoint 协议。"""

import importlib
import sys
import types

import pytest
import torch

from test_mm_mid_io import config, inference_parser, sample, train_helpers, training_parser
from test_mm_ucpmamba_novmamba import (
    mid_module as _mid_fixture,
    plain_module as _plain_fixture,
    real_convlstm as _recurrence_fixture,
)


ALIASES = ("mm-ucpmamba-novmamba", "mm_ucpmamba_novmamba")

mid_module = _mid_fixture
plain_module = _plain_fixture
real_convlstm = _recurrence_fixture


@pytest.mark.parametrize("name", ALIASES)
def test_train_and_inference_accept_both_model_spellings(name):
    train = training_parser()
    inference, _ = inference_parser()
    assert train.parse_args(["--model", name]).model == name
    assert inference.parse_args(["--model", name]).model == name


@pytest.mark.parametrize("name", ALIASES)
def test_model_factory_selects_plain_async_constructor(plain_module, monkeypatch, name):
    module, _, _, _ = plain_module
    model_utils = importlib.import_module("src.model_utils")
    received = {}

    def constructor(**kwargs):
        received.update(kwargs)
        return "plain async"

    monkeypatch.setattr(module, "MMUCPMambaNoVMamba", constructor)
    cfg = config(name)
    cfg.__dict__.update(encoder_widths=[8, 8, 8, 16], decoder_widths=[8, 8, 8, 16],
                        out_conv=[8, 4], str_conv_k=4, str_conv_s=2, str_conv_p=1,
                        encoder_norm="group", decoder_norm="group", decoder_groups=8,
                        padding_mode="reflect", pad_value=0)
    assert model_utils.get_model(cfg) == "plain async"
    assert received["hidden_dim"] == 64
    assert received["dynamic_enabled"] and received["interaction_enabled"]
    assert received["decoder_norm"] == "group"
    assert (received["num_bins"], received["start_day"], received["end_day"]) == (24, 0, 40)


@pytest.mark.parametrize("name", ALIASES)
def test_forward_keeps_each_modality_date_axis_and_mask(name):
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
def test_metadata_only_adds_plain_temporal_encoder_and_separate_result_path(name):
    metadata = train_helpers()["_set_mm_experiment_metadata"]
    baseline, new = config("mm_ucpmamba_async"), config(name)
    for cfg in (baseline, new):
        metadata(cfg)
    assert new.res_dir != baseline.res_dir
    assert new.experiment_name == "mm_ucpmamba_novmamba_s2_s1a_s1d"
    assert new.active_modalities == baseline.active_modalities == ["S2", "S1A", "S1D"]
    assert new.mid_alignment == baseline.mid_alignment
    assert new.mamba_variant is None and baseline.mamba_variant is not None
    assert new.async_design == dict(baseline.async_design, temporal_encoder="plain_convlstm")
    assert "temporal_encoder" not in baseline.async_design


def test_last_checkpoint_restores_alias_and_rejects_other_temporal_models(tmp_path):
    from src.mm_async_experiment import check_async_checkpoint, configure_async, save_async_last

    cfg = config(ALIASES[0])
    configure_async(cfg)
    cfg.res_dir = str(tmp_path)
    (tmp_path / "Fold_2").mkdir()
    module = torch.nn.Conv2d(1, 1, 1)
    optimizer = torch.optim.Adam(module.parameters())
    save_async_last(module, optimizer, cfg, fold=2, epoch=3, best_miou=.6)
    saved = torch.load(tmp_path / "Fold_2/last.pth.tar", weights_only=True)
    assert saved["model_metadata"]["model_name"] == ALIASES[0]
    assert saved["model_metadata"]["async_design"]["temporal_encoder"] == "plain_convlstm"
    assert saved["epoch"] == 3 and saved["best_val_miou"] == .6
    torch.testing.assert_close(saved["state_dict"]["weight"], module.weight)
    check_async_checkpoint(saved, cfg)
    check_async_checkpoint(saved, config(ALIASES[1]))
    other = config("mm_ucpmamba_async")
    with pytest.raises(ValueError, match="configuration/reference date mismatch"):
        check_async_checkpoint(saved, other)
    configure_async(other)
    with pytest.raises(ValueError, match="configuration/reference date mismatch"):
        check_async_checkpoint({"model_metadata": {"async_design": other.async_design}}, cfg)
    changed_calendar = config(ALIASES[1])
    changed_calendar.ref_date = "2019-09-01"
    with pytest.raises(ValueError, match="configuration/reference date mismatch"):
        check_async_checkpoint(saved, changed_calendar)


def test_entrypoint_changes_only_model_default_and_keeps_parser_unchanged(monkeypatch):
    parser = training_parser()
    train = types.ModuleType("train_semantic")
    train.parser, train.list_args, train.main = parser, ["encoder_widths", "decoder_widths", "out_conv"], None
    monkeypatch.setitem(sys.modules, "train_semantic", train)
    for module_name in ("run_mm_ucpmamba_async", "run_mm_ucpmamba_novmamba"):
        monkeypatch.delitem(sys.modules, module_name, raising=False)
    try:
        original = importlib.import_module("run_mm_ucpmamba_async")
        entrypoint = importlib.import_module("run_mm_ucpmamba_novmamba")
        default_model = parser.get_default("model")
        old, new = original.make_config([]), entrypoint.make_config([])
        assert old.model == "mm_ucpmamba_async" and new.model == ALIASES[0]
        assert parser.get_default("model") == default_model
        assert {key for key in vars(old) if getattr(old, key) != getattr(new, key)} == {"model"}
        assert new.encoder_widths == [64, 64, 64, 128]
        assert new.decoder_norm == "group" and new.mid_low_memory
        assert (new.mid_num_bins, new.mid_start_day, new.mid_end_day) == (24, -200., 600.)
        assert new.ignore_index == 19 and new.epochs == 100
        assert entrypoint.make_config(["--model", ALIASES[1]]).model == ALIASES[1]
    finally:
        for module_name in ("run_mm_ucpmamba_async", "run_mm_ucpmamba_novmamba"):
            sys.modules.pop(module_name, None)
