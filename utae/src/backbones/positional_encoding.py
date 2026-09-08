import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    def __init__(self, d, T=1000, repeat=None, offset=0):
        # d: 输出的特征维度。
        # T: 控制频率缩放的常数（默认值为 1000），对应位置编码公式中的基数。
        # repeat: 如果指定，将生成的编码在最后一个维度重复 repeat 次。
        # offset: 偏移量，用于调整编码的起始位置，影响生成的位置编码。
        super(PositionalEncoder, self).__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        self.denom = torch.pow(
            T, 2 * (torch.arange(offset, offset + d).float() // 2) / d
        )
        # 生成从 offset 到 offset + d 的整数序列。
        self.updated_location = False
    #     标记 denom 是否已移动到与输入相同的设备（如 GPU）。避免每次 forward 调用时重复执行。

    def forward(self, batch_positions):
        # batch_positions: 输入的位置序列，形状为 (𝐵,𝑇)其中B是batch大小，T是序列长度。
        if not self.updated_location:
            self.denom = self.denom.to(batch_positions.device)
            # 确保 denom 的设备（如 GPU/CPU）与输入数据一致。
            self.updated_location = True
        sinusoid_table = (
            batch_positions[:, :, None] / self.denom[None, None, :]
        )  # B x T x C

        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # dim 2i
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1

        if self.repeat is not None:
            sinusoid_table = torch.cat(
                [sinusoid_table for _ in range(self.repeat)], dim=-1
            )

        return sinusoid_table
