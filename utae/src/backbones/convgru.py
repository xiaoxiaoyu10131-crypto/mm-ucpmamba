"""
Modified from https://github.com/TUM-LMF/MTLCC-pytorch/blob/master/src/models/convlstm/convlstm.py
authors: TUM-LMF
"""
import torch.nn as nn
from torch.autograd import Variable
import torch


class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvGRUCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # 用于计算更新门Z和重置门R
        # 输入：输入张量和当前隐藏状态的拼接结果  输出：形状为2*hidden_dim的特征图
        self.in_conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=2 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )
        # 用于计算候选隐藏状态h
        # 输入：输入张量和重置后的隐藏状态（r * cur_state）的拼接结果。
        # 输出：形状为 hidden_dim 的特征图。
        self.out_conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        # input_tensor：当前时间步的输入张量，形状为 (batch_size, input_dim, height, width)。
        # cur_state：当前时间步的隐藏状态，形状为 (batch_size, hidden_dim, height, width)
        combined = torch.cat([input_tensor, cur_state], dim=1)
        z, r = torch.sigmoid(self.in_conv(combined)).chunk(2, dim=1)
        # 将拼接结果输入到 self.in_conv，输出2*hidden_dim通道的特征图并按通道维度切为两个部分
        # 更新门（z）：决定当前隐藏状态和候选隐藏状态的混合比例。
        # 重置门（r）：决定当前隐藏状态中有多少信息参与计算候选隐藏状态。
        h = torch.tanh(self.out_conv(torch.cat([input_tensor, r * cur_state], dim=1)))
        # 将 cur_state 与 r 相乘，得到重置后的隐藏状态。
        # 将 input_tensor 和重置后的隐藏状态拼接，输入到 self.out_conv，得到候选隐藏状态 h_tilde
        new_state = (1 - z) * cur_state + z * h
        # 输出形状为 (batch_size, hidden_dim, height, width)。
        return new_state

    def init_hidden(self, batch_size, device):
        return Variable(
            torch.zeros(batch_size, self.hidden_dim, self.height, self.width)
        ).to(device)


class ConvGRU(nn.Module):
    def __init__(
        self,
        input_size,
        input_dim,
        hidden_dim,
        kernel_size,
        num_layers=1,
        batch_first=True,
        bias=True,
        return_all_layers=False,
    ):
        super(ConvGRU, self).__init__()

        self._check_kernel_size_consistency(kernel_size)
        # _check_kernel_size_consistency 确保 kernel_size 是单个元组或多元组列表

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length.")

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(
                ConvGRUCell(
                    input_size=(self.height, self.width),
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(
        self, input_tensor, hidden_state=None, pad_mask=None, batch_positions=None
    ):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        pad_maks (b , t)
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(
                batch_size=input_tensor.size(0), device=input_tensor.device
            )

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :], cur_state=h
                )
                output_inner.append(h)
            #     输出每个时间步的隐藏状态，并将其存入 output_inner

            layer_output = torch.stack(output_inner, dim=1)
            # 堆叠时间步的输出，形成层的输出张量。
            if pad_mask is not None:
                last_positions = (~pad_mask).sum(dim=1) - 1
                layer_output = layer_output[:, last_positions, :, :, :]
            # 如果有填充掩码 pad_mask，则仅保留最后有效时间步的输出。

            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append(h)

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1]
            last_state_list = last_state_list[-1]

        return layer_output_list, last_state_list
    # 默认返回最后一层的输出和隐藏状态

    # 为每层初始化隐藏状态，形状为 (batch_size, hidden_dim, height, width)
    def _init_hidden(self, batch_size, device):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, device))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (
            isinstance(kernel_size, tuple)
            or (
                isinstance(kernel_size, list)
                and all([isinstance(elem, tuple) for elem in kernel_size])
            )
        ):
            raise ValueError("`kernel_size` must be tuple or list of tuples")

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ConvGRU_Seg(nn.Module):
    def __init__(
        self, num_classes, input_size, input_dim, hidden_dim, kernel_size, pad_value=0
    ):
        super(ConvGRU_Seg, self).__init__()
        self.convgru_encoder = ConvGRU(
            input_dim=input_dim,
            input_size=input_size,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            return_all_layers=False,
        )
        self.classification_layer = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=num_classes,
            kernel_size=kernel_size,
            padding=1,
        )
        self.pad_value = pad_value

    def forward(self, input, batch_positions=None):
        pad_mask = (
            (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
        )  # BxT pad mask
        pad_mask = pad_mask if pad_mask.any() else None
        _, out = self.convgru_encoder(input, pad_mask=pad_mask)
        out = self.classification_layer(out)
        return out
