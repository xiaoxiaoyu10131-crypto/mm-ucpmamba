"""Original UConvLSTM components kept separate from UCPMamba."""

import torch
import torch.nn as nn

from src.backbones.convlstm import ConvLSTM
from src.backbones.utae import RecUNet


class PlainConvLSTMCell(nn.Module):
    """Standard ConvLSTM cell without the UCPMamba MSVSS block."""

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        super().__init__()
        self.height, self.width = input_size
        self.hidden_dim = hidden_dim
        padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        gates = self.conv(torch.cat([input_tensor, h_cur], dim=1))
        cc_i, cc_f, cc_o, cc_g = torch.split(gates, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        return o * torch.tanh(c_next), c_next

    def init_hidden(self, batch_size, device):
        state = torch.zeros(
            batch_size, self.hidden_dim, self.height, self.width, device=device
        )
        return state.clone(), state.clone()


class PlainConvLSTM(ConvLSTM):
    """ConvLSTM with the shared, padding-safe recurrence implementation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, cell_class=PlainConvLSTMCell, **kwargs)


class UConvLSTM(RecUNet):
    """Original RecUNet: plain ConvLSTM and temporal mean skip aggregation."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("temporal_encoder_class", PlainConvLSTM)
        kwargs.setdefault("skip_aggregation", "mean")
        kwargs.setdefault("decoder_norm", kwargs.get("encoder_norm", "group"))
        super().__init__(*args, **kwargs)
