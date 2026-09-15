import pytest
import torch

from src.temporal_dropout import apply_temporal_dropout


def _sample():
    lengths = {"S2": 5, "S1A": 7, "S1D": 9}
    data = {
        key: torch.arange(length).view(length, 1, 1, 1).float()
        for key, length in lengths.items()
    }
    dates = {
        key: torch.arange(length) + offset
        for (key, length), offset in zip(lengths.items(), (100, 200, 300))
    }
    return data, dates


def test_temporal_dropout_keeps_images_and_dates_synchronized():
    torch.manual_seed(7)
    data, dates = _sample()
    filtered_data, filtered_dates = apply_temporal_dropout(
        data, dates, {"S2": 0.4, "S1A": 0.2, "S1D": 0.2}
    )

    for modality, offset in (("S2", 100), ("S1A", 200), ("S1D", 300)):
        frame_ids = filtered_data[modality][:, 0, 0, 0].long()
        assert filtered_data[modality].shape[0] >= 1
        assert torch.equal(filtered_dates[modality], frame_ids + offset)


def test_temporal_dropout_always_keeps_one_frame_per_modality():
    torch.manual_seed(3)
    data, dates = _sample()
    filtered_data, filtered_dates = apply_temporal_dropout(
        data, dates, {"S2": 1.0, "S1A": 1.0, "S1D": 1.0}
    )

    assert all(value.shape[0] == 1 for value in filtered_data.values())
    assert all(value.shape[0] == 1 for value in filtered_dates.values())


def test_temporal_dropout_rejects_misaligned_dates():
    data, dates = _sample()
    dates["S2"] = dates["S2"][:-1]

    with pytest.raises(ValueError, match="5 frames but 4 dates"):
        apply_temporal_dropout(data, dates, {"S2": 0.4})
