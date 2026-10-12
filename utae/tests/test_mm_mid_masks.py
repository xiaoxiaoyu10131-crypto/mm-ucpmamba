"""针对缺测、稀疏掩码、半精度与日期配置加载风险的回归测试。"""

import importlib

import pytest
import torch
import torch.nn as nn

from test_mm_temporal_alignment import aligner_class as _aligner_fixture
from test_mm_ucpmamba_mid import dates, inputs, mid_module as _mid_fixture, model


aligner_class, mid_module = _aligner_fixture, _mid_fixture


def test_sparse_one_pixel_survives_spatial_mask_downsampling(mid_module):
    instance = model(mid_module).eval()
    x = inputs()
    masks = {}
    for name, value in zip(("S2", "S1A", "S1D"), x):
        mask = torch.zeros(2, value.shape[1], 1, 16, 16, dtype=torch.bool)
        mask[:, 0, :, -1, -1] = True
        masks[name] = mask
    with torch.no_grad():
        debug = instance(*x, **dates(), valid_masks=masks, return_debug=True)
    for name in masks:
        assert debug["aligned_valid"][name].any()
        assert debug["aligned_counts"][name].sum() == 2
    assert debug["pixel_valid"].any()
    assert not debug["temporal_pad_mask"].all()
    assert torch.isfinite(debug["logits"]).all()


@pytest.mark.parametrize("bad_date", [40., float("nan")])
def test_dropout_cannot_hide_invalid_observation_date(mid_module, bad_date):
    instance = model(mid_module, modality_dropout=1.0).train()
    positions = dates()
    positions["batch_positions_s1a"][:, 0] = bad_date
    with pytest.raises(ValueError):
        instance(*inputs(), **positions)


def test_half_precision_phenology_mean_does_not_overflow(mid_module):
    components = importlib.import_module("src.backbones.mm_mid_components")
    aggregator = mid_module.PhenologyAttentionV2(8, stage_level=2).half()
    for layer in aggregator.modules():
        if isinstance(layer, nn.Conv3d):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    sequence = torch.full((2, 2, 8, 2, 2), 60000., dtype=torch.float16)
    valid = torch.ones(2, 2, 1, 2, 2, dtype=torch.bool)
    with torch.no_grad():
        result = components.masked_phenology(aggregator, sequence, valid)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result.float(), torch.full_like(result.float(), 60000.), rtol=0, atol=32)


def test_dropout_keeps_a_real_available_modality(mid_module):
    instance = model(mid_module, modality_dropout=1.0).train()
    x = inputs()
    # S2 确实缺测；S1A 与 S1D 有数据。不能为防止全丢而重新启用 S2。
    masks = {"S2": torch.zeros(2, 3, dtype=torch.bool)}
    debug = instance(*x, **dates(), valid_masks=masks, return_debug=True)
    assert not debug["aligned_valid"]["S2"].any()
    kept = torch.stack([debug["aligned_valid"][name].flatten(1).any(dim=1)
                        for name in ("S1A", "S1D")]).sum(dim=0)
    assert kept.tolist() == [1, 1]
    assert torch.isfinite(debug["logits"]).all()


def test_dropout_does_not_create_observations_for_all_missing_sample(mid_module):
    instance = model(mid_module, modality_dropout=1.0).train()
    x = inputs()
    masks = {name: torch.zeros(2, value.shape[1], dtype=torch.bool)
             for name, value in zip(("S2", "S1A", "S1D"), x)}
    debug = instance(*x, **dates(), valid_masks=masks, return_debug=True)
    assert debug["window_pad_mask"].all() and debug["temporal_pad_mask"].all()
    assert not debug["pixel_valid"].any()
    assert debug["fused_sequence"].count_nonzero() == 0
    assert torch.isfinite(debug["logits"]).all()


def test_state_dict_roundtrip_preserves_alignment_calendar(aligner_class):
    first = aligner_class(num_bins=4, start_day=0, end_day=40)
    second = aligner_class(num_bins=4, start_day=0, end_day=40)
    second.load_state_dict(first.state_dict())
    values = torch.tensor([3., 7.]).view(1, 2, 1, 1, 1)
    positions, pad = torch.tensor([[1., 21.]]), torch.zeros(1, 2, dtype=torch.bool)
    result = first(values, positions, pad)
    restored = second(values, positions, pad)
    for original, loaded in zip(result, restored):
        torch.testing.assert_close(original, loaded)


def test_half_precision_date_window_mean_does_not_overflow(aligner_class):
    aligner = aligner_class(num_bins=2, start_day=0, end_day=20)
    values = torch.full((1, 2, 1, 1, 1), 60000., dtype=torch.float16)
    pooled, _, _ = aligner(values, torch.tensor([[1., 3.]]),
                           torch.zeros(1, 2, dtype=torch.bool))
    assert torch.isfinite(pooled).all()
    assert pooled[0, 0, 0, 0, 0].item() == 60000.


@pytest.mark.parametrize("strict", [False, True])
def test_state_dict_rejects_different_calendar_even_with_equal_k(aligner_class, strict):
    source = aligner_class(num_bins=4, start_day=0, end_day=40)
    target = aligner_class(num_bins=4, start_day=10, end_day=50)
    with pytest.raises((ValueError, RuntimeError)):
        target.load_state_dict(source.state_dict(), strict=strict)
    assert (target.edges[0].item(), target.edges[-1].item()) == (10., 50.)
