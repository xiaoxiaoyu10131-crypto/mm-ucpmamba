"""仅供新 mid 模型使用：按真实序列长度构造显式 padding 掩码。"""

import torch

from src.backbones.mm_temporal_alignment import normalize_valid_mask
from src.utils import pad_collate


def mid_pad_collate(batch, pad_value=0):
    """输入沿用 ((data_dict, dates_dict), target)，允许 data 内附带掩码字典。

    pad_masks 的 True 表示 padding；valid_masks 的 True 表示有效观测。
    自动补齐的帧一定标记为 padding，不通过影像是否全零判断。
    """
    normalized = []
    with_valid = any("valid_masks" in sample[0][0] for sample in batch)
    for (data, dates), target in batch:
        if not isinstance(data, dict) or not isinstance(dates, dict):
            raise TypeError("mid_pad_collate requires multimodal data and date dictionaries")
        prepared = dict(data)
        pads, validities = {}, {}
        for name in ("S2", "S1A", "S1D"):
            if name not in data:
                continue
            sequence = data[name]
            if sequence.ndim != 4 or name not in dates or dates[name].ndim != 1:
                raise ValueError("each sample needs [T,C,H,W] data and [T] dates")
            if sequence.shape[0] != dates[name].shape[0]:
                raise ValueError(f"{name} observations and dates have different lengths")
            explicit = data.get("pad_masks", {}).get(name)
            if explicit is not None and (
                explicit.dtype != torch.bool or explicit.shape != sequence.shape[:1]
            ):
                raise ValueError("sample pad_masks must be bool with shape [T]")
            pads[name] = (
                explicit
                if explicit is not None
                else torch.zeros(sequence.shape[0], dtype=torch.bool, device=sequence.device)
            )
            if with_valid:
                validity = data.get("valid_masks", {}).get(name)
                validities[name] = normalize_valid_mask(
                    sequence[None],
                    pads[name][None],
                    None if validity is None else validity[None],
                )[0]
        prepared["pad_masks"] = pads
        if with_valid:
            prepared["valid_masks"] = validities
        normalized.append(((prepared, dates), target))
    collated = pad_collate(normalized, pad_value=pad_value)
    data = collated[0][0]
    explicit = data.get("pad_masks", {})
    masks = {}
    for name in ("S2", "S1A", "S1D"):
        if name not in data:
            continue
        sequence = data[name]
        lengths = torch.tensor(
            [sample[0][0][name].shape[0] for sample in batch], device=sequence.device
        )
        masks[name] = (
            torch.arange(sequence.shape[1], device=sequence.device)[None] >= lengths[:, None]
        )
        if name in explicit:
            masks[name] = masks[name] | explicit[name]
    data["pad_masks"] = masks
    return collated
