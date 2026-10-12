"""Date-window alignment checks on real tensors; no CUDA backend is needed."""

import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture
def aligner_class():
    path = Path(__file__).resolve().parents[1] / "src/backbones/mm_temporal_alignment.py"
    spec = importlib.util.spec_from_file_location("_mid_alignment_test_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DateWindowAligner


def test_dates_define_windows_instead_of_frame_index(aligner_class):
    aligner = aligner_class(num_bins=4, start_day=0, end_day=40)
    sequence = torch.tensor([2., 8., 12., 24., 36.]).view(1, 5, 1, 1, 1)
    dates = torch.tensor([[1., 9., 19., 20., 39.]])
    pooled, valid, counts = aligner(sequence, dates, torch.zeros(1, 5, dtype=torch.bool))
    assert pooled.flatten().tolist() == [5., 12., 24., 36.]
    assert counts.flatten().tolist() == [2, 1, 1, 1]
    assert counts.dtype == torch.int64
    assert valid.shape == counts.shape == (1, 4, 1, 1, 1)
    assert valid.all()


def test_unsorted_dates_and_different_sequences_share_calendar(aligner_class):
    aligner = aligner_class(num_bins=4, start_day=0, end_day=40)
    pad = torch.zeros(1, 3, dtype=torch.bool)
    first, valid, _ = aligner(
        torch.tensor([3., 1., 5.]).view(1, 3, 1, 1, 1),
        torch.tensor([[25., 5., 29.]]), pad,
    )
    second, _, _ = aligner(
        torch.tensor([7., 9.]).view(1, 2, 1, 1, 1),
        torch.tensor([[6., 24.]]), torch.zeros(1, 2, dtype=torch.bool),
    )
    assert first.flatten().tolist() == [1., 0., 4., 0.]
    assert second.flatten().tolist() == [7., 0., 9., 0.]
    assert valid.flatten().tolist() == [True, False, True, False]


def test_pixel_validity_excludes_cloud_and_nonfinite_invalid_values(aligner_class):
    aligner = aligner_class(num_bins=2, start_day=0, end_day=20)
    sequence = torch.tensor([2., float("nan"), 6., 8.]).view(1, 2, 1, 1, 2)
    valid_mask = torch.tensor([True, False, True, True]).view(1, 2, 1, 1, 2)
    pooled, valid, counts = aligner(
        sequence, torch.tensor([[1., 2.]]),
        torch.zeros(1, 2, dtype=torch.bool), valid_mask=valid_mask,
    )
    assert pooled[0, 0, 0, 0].tolist() == [4., 8.]
    assert counts[0, 0, 0, 0].tolist() == [2, 1]
    assert pooled[:, 1].count_nonzero() == 0
    assert not valid[:, 1].any()
    assert torch.isfinite(pooled).all()


def test_frame_masks_and_padding_ignore_invalid_dates(aligner_class):
    aligner = aligner_class(num_bins=2, start_day=0, end_day=20)
    sequence = torch.tensor([3., float("inf"), float("nan")]).view(1, 3, 1, 1, 1)
    pooled, _, counts = aligner(
        sequence, torch.tensor([[0., float("nan"), 99.]]),
        torch.tensor([[False, True, False]]),
        valid_mask=torch.tensor([[True, True, False]]),
    )
    assert pooled.flatten().tolist() == [3., 0.]
    assert counts.flatten().tolist() == [1, 0]


@pytest.mark.parametrize("date", [-1., 20., float("nan"), float("inf")])
def test_active_dates_outside_left_closed_right_open_range_raise(aligner_class, date):
    aligner = aligner_class(num_bins=2, start_day=0, end_day=20)
    with pytest.raises(ValueError):
        aligner(torch.ones(1, 1, 1, 1, 1), torch.tensor([[date]]),
                torch.zeros(1, 1, dtype=torch.bool))


@pytest.mark.parametrize("length", [0, 3])
def test_all_missing_and_empty_sequences_are_strict_zero(aligner_class, length):
    aligner = aligner_class(num_bins=3, start_day=0, end_day=30)
    pooled, valid, counts = aligner(
        torch.full((2, length, 2, 2, 2), float("nan")),
        torch.full((2, length), float("nan")),
        torch.ones(2, length, dtype=torch.bool),
    )
    assert pooled.shape == (2, 3, 2, 2, 2)
    assert torch.isfinite(pooled).all()
    assert pooled.count_nonzero() == counts.count_nonzero() == 0
    assert not valid.any()


def test_gradient_only_reaches_valid_observations(aligner_class):
    aligner = aligner_class(num_bins=2, start_day=0, end_day=20)
    sequence = torch.tensor([2., 4., 9.]).view(1, 3, 1, 1, 1).requires_grad_()
    pooled, _, _ = aligner(
        sequence, torch.tensor([[1., 3., 15.]]),
        torch.zeros(1, 3, dtype=torch.bool),
        valid_mask=torch.tensor([[True, True, False]]),
    )
    pooled.sum().backward()
    assert sequence.grad.flatten().tolist() == [0.5, 0.5, 0.]


@pytest.mark.parametrize("kwargs", [
    {"num_bins": 0, "start_day": 0, "end_day": 20},
    {"num_bins": 2, "start_day": 20, "end_day": 0},
    {"num_bins": 2, "start_day": None, "end_day": None},
])
def test_bad_window_configuration_rejected(aligner_class, kwargs):
    with pytest.raises((ValueError, TypeError)):
        aligner_class(**kwargs)


def test_bad_mask_shape_and_dtype_rejected(aligner_class):
    aligner = aligner_class(num_bins=2, start_day=0, end_day=20)
    sequence, dates = torch.ones(1, 2, 3, 2, 2), torch.tensor([[1., 2.]])
    pad = torch.zeros(1, 2, dtype=torch.bool)
    with pytest.raises(ValueError):
        aligner(sequence, dates, torch.zeros(1, 1, dtype=torch.bool))
    with pytest.raises((ValueError, TypeError)):
        aligner(sequence, dates, pad, valid_mask=torch.ones(1, 2))
    with pytest.raises(ValueError):
        aligner(sequence, dates, pad, valid_mask=torch.ones(1, 2, 3, 2, 2, dtype=torch.bool))
