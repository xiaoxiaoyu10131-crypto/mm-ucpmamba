"""Async 按真实日期同步排序影像与显式掩码，再复用 mid padding。"""

import torch

from src.mm_collate import mid_pad_collate


def async_pad_collate(batch, pad_value=0):
    sorted_batch = []
    for (images, dates), target in batch:
        data, times = dict(images), dict(dates)
        for key in ("pad_masks", "valid_masks"):
            if key in images:
                data[key] = dict(images[key])
        for name in ("S2", "S1A", "S1D"):
            if name not in images:
                continue
            if dates[name].ndim != 1 or len(images[name]) != len(dates[name]):
                raise ValueError("async observations and dates must have matching lengths")
            order = torch.argsort(dates[name], stable=True)
            data[name], times[name] = images[name][order], dates[name][order]
            for key in ("pad_masks", "valid_masks"):
                if name in data.get(key, {}):
                    data[key][name] = images[key][name][order]
        sorted_batch.append(((data, times), target))
    return mid_pad_collate(sorted_batch, pad_value=pad_value)
