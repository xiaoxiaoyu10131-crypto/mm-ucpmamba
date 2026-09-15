import torch


def apply_temporal_dropout(data, dates, probabilities):
    """Drop frames per modality while keeping image/date indices aligned."""
    filtered_data = {}
    filtered_dates = {}
    #对齐校验（第 8–18 行）：取出该模态的影像帧 frames 和对应日期 dates，检查数量是否一致，且至少要有一帧，否则报错
    for modality, frames in data.items():
        modality_dates = dates[modality]
        if modality_dates.ndim == 0:
            modality_dates = modality_dates.unsqueeze(0)
        if frames.shape[0] != modality_dates.shape[0]:
            raise ValueError(
                f"{modality} has {frames.shape[0]} frames but "
                f"{modality_dates.shape[0]} dates"
            )
        if frames.shape[0] == 0:
            raise ValueError(f"{modality} must contain at least one frame")
        #读取丢弃率（第 20–24 行）：从 probabilities 字典里取该模态的概率（如 {"S2": 0.1, ...}），没配就默认 0（不丢），并校验在 0~1 之间。
        probability = float(probabilities.get(modality, 0.0))
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"Temporal dropout for {modality} must be between 0 and 1"
            )
        #随机抽样（第 26–28 行）：为每一帧生成一个随机数，>= probability 的帧保留。如果运气不好全部被丢掉，就强制随机保留 1 帧，保证序列不为空（第 27–28 行的兜底逻辑）。
        keep = torch.rand(frames.shape[0]) >= probability
        if not keep.any():
            keep[torch.randint(frames.shape[0], size=(1,)).item()] = True
        indices = keep.nonzero(as_tuple=False).squeeze(1)
        #同步筛选（第 29–31 行）：用保留帧的索引同时对 frames 和 dates 做 index_select，这样丢掉第 i 帧时，第 i 个日期也会一起被丢掉，两者始终对齐。
        filtered_data[modality] = frames.index_select(0, indices)
        filtered_dates[modality] = modality_dates.index_select(0, indices)

    return filtered_data, filtered_dates
