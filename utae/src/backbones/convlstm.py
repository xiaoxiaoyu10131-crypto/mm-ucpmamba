"""
Taken from https://github.com/TUM-LMF/MTLCC-pytorch/blob/master/src/models/convlstm/convlstm.py
authors: TUM-LMF
"""
import torch.nn as nn
from torch.autograd import Variable
import torch
from src.backbones.vmamba.vmamba import VSSBlock, LayerNorm2d, Linear2d
from typing import Sequence, Type, Optional, Union
# from src.backbones.Bas import BasicLayer


class MSConv(nn.Module):
    def __init__(self, dim: int, kernel_sizes: Sequence[int] = (1, 3, 5)) -> None:
        super(MSConv, self).__init__()
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, bias=False)
            for kernel_size in kernel_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + sum([conv(x) for conv in self.dw_convs])

class MS_MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.,
        channels_first: bool = False,
        kernel_sizes: Sequence[int] = (1, 3, 5),
    ) -> None:
        super(MS_MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = Linear2d if channels_first else nn.Linear

        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.multiscale_conv = MSConv(hidden_features, kernel_sizes=kernel_sizes)
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.multiscale_conv(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MSVSS(nn.Sequential):
    def __init__(
            self,
            dim: int,
            drop_path: Union[Sequence[float], float] = 0.0,
            use_checkpoint: bool = False,
            norm_layer: Type[nn.Module] = LayerNorm2d,
            channel_first: bool = True,
            ssm_d_state: int = 16,
            ssm_ratio: float = 1.0,
            ssm_dt_rank: str = "auto",
            ssm_act_layer: Type[nn.Module] = nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias: bool = False,
            ssm_drop_rate: float = 0.01,
            ssm_init: str = "v0",
            forward_type: str = "v01_noz",
            ssm_core: str = "ss2d",
            mlp_ratio: float = 4.0,
            mlp_act_layer: Type[nn.Module] = nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp: bool = False,
    ) -> None:
        # 构建一个单层的 VSSBlock
        block = VSSBlock(
            hidden_dim=dim,
            drop_path=drop_path,
            norm_layer=norm_layer,
            channel_first=channel_first,
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_dt_rank=ssm_dt_rank,
            ssm_act_layer=ssm_act_layer,
            ssm_conv=ssm_conv,
            ssm_conv_bias=ssm_conv_bias,
            ssm_drop_rate=ssm_drop_rate,
            ssm_init=ssm_init,
            forward_type=forward_type,
            ssm_core=ssm_core,
            mlp_ratio=mlp_ratio,
            mlp_act_layer=mlp_act_layer,
            mlp_drop_rate=mlp_drop_rate,
            gmlp=gmlp,
            use_checkpoint=use_checkpoint,
            customized_mlp=MS_MLP
        )

        # 使用 nn.Sequential 进行封装
        super(MSVSS, self).__init__(block)


VMAMBA_V2_SSM_KWARGS = {
    "ssm_d_state": 1,
    "ssm_ratio": 2.0,
    "forward_type": "v05_noz",
}

MAMBA2_SSM_KWARGS = {
    "ssm_core": "mamba2",
    "ssm_d_state": 64,
    "ssm_ratio": 1.0,
    "ssm_act_layer": nn.GELU,
    "ssm_init": "v2",
    "forward_type": "m0_noz",
}


class ConvLSTMCell(nn.Module):
    def __init__(
        self, input_size, input_dim, hidden_dim, kernel_size, bias,
        vmamba_mode="g_only", vss_variant="vmamba1",
    ):
        """
        初始化convlstm单元.

        参数
        ----------
        input_size: (int, int)
            输入张量的高度和宽度 (height, width).
        input_dim: int
            输入张量的通道数.
        hidden_dim: int
            N隐藏状态的通道数.
        kernel_size: (int, int)
            卷积核大小.
        bias: bool
            是否添加偏置项.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.vmamba_mode = vmamba_mode
        self.vss_variant = vss_variant

        variant_kwargs = {
            "vmamba1": {},
            "vmambav2": VMAMBA_V2_SSM_KWARGS,
            "mamba2": MAMBA2_SSM_KWARGS,
        }
        if self.vss_variant not in variant_kwargs:
            raise ValueError(f"Unsupported VSS variant: {self.vss_variant}")
        vss_kwargs = variant_kwargs[self.vss_variant]

        # 初始化卷积层，用于计算4个门
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        if self.vmamba_mode == "direct":
            self.vss = MSVSS(
                dim=4 * self.hidden_dim, drop_path=0.01, **vss_kwargs
            )
        elif self.vmamba_mode == "g_only":
            # Apply VMamba only to the candidate-state branch (g), not to the
            # input/forget/output gates. The branch starts as an exact identity.
            self.vss = MSVSS(dim=self.hidden_dim, drop_path=0.01, **vss_kwargs)
            self.beta = nn.Parameter(torch.zeros(1))
        else:
            raise ValueError(f"Unsupported vmamba_mode: {self.vmamba_mode}")
        # self.sat = BasicLayer(num_layers=1, dim=[256, 256], n_iter=1, stoken_size=[1, 1], num_heads=4)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        # input_tensor为当前时间步的输入张量，形状为(B,C,H,W)
        # 当前隐藏状态和细胞状态，格式为(h_cur,c_cur)

        combined = torch.cat(
            [input_tensor, h_cur], dim=1
        )
        # 在通道维度拼接输入和当前隐藏状态

        combined_conv = self.conv(combined)
        # print("111111111111111111111111111111111111111111111")
        # print(combined_conv.shape)

        if self.vmamba_mode == "direct":
            combined_conv = self.vss(combined_conv)
            cc_i, cc_f, cc_o, cc_g = torch.split(
                combined_conv, self.hidden_dim, dim=1
            )
        else:
            cc_i, cc_f, cc_o, cc_g = torch.split(
                combined_conv, self.hidden_dim, dim=1
            )
            g_enhanced = self.vss(cc_g)
            cc_g = cc_g + self.beta * (g_enhanced - cc_g)
        # print("1111111111111111111111111111111111111111111112222222222222222222222222")
        # print(combined_conv.shape)

        # 将卷积输出分为4部分，分别对应4个门
        # 输入门、遗忘门、输出门和候选细胞状态
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # 更新细胞状态和隐藏状态
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next
    #     返回下一时间步的隐藏状态和下一时间步的细胞状态

    def init_hidden(self, batch_size, device):
        # 初始化隐藏状态和细胞状态为零
        # 参数batch_size为批量大小 ， device为cpu或者是cuda
        # 返回隐藏状态和细胞状态，形状分别为(B,hidden_dim,H,W)
        return (
            Variable(
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width)
            ).to(device),
            Variable(
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width)
            ).to(device),
        )


class DirectVSSConvLSTMCell(ConvLSTMCell):
    """Original UCPMamba cell with direct four-gate VSS enhancement."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, vmamba_mode="direct", **kwargs)


class VMambaV2ConvLSTMCell(ConvLSTMCell):
    """Candidate-state ConvLSTM cell using the official VMamba v2 settings."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, vss_variant="vmambav2", **kwargs)


class Mamba2ConvLSTMCell(ConvLSTMCell):
    """Candidate-state ConvLSTM cell using the true Mamba2/SSD core."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, vss_variant="mamba2", **kwargs)


class ConvLSTM(nn.Module):
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
        cell_class=ConvLSTMCell,
    ):
        # input_size:输入的高度和宽度  input_dim；输入通道数
        # hidden_dim：隐藏层的通道数，可以为单值或列表（每层不同）。
        # num_layers：ConvLSTM 的层数。batch_first：指定输入是否以 (batch, seq, ...) 为顺序
        # return_all_layers：是否返回所有层的输出

        super(ConvLSTM, self).__init__()
        # 调用父类nn.module的构造函数

        self._check_kernel_size_consistency(kernel_size)
        # 检查kernel_size是否符合要求，必须是元组和元组列表

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        # 如果kernel_size或hidden_dim是单值，通过_extend_for_multilayer 转换为与层数一致的列表。
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length.")
        # 确保 kernel_size 和 hidden_dim 的长度与 num_layers 一致，否则抛出异常。

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        # 初始化 cell_list：存储每一层的 ConvLSTMCell
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            # 对于每一层的输入通道数等于input_dim,后续层的输入通道数等于前一层的隐藏通道数

            cell_list.append(
                cell_class(
                    input_size=(self.height, self.width),
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                )
            )
        #     创建每一层的ConvLSTMCell,并加入cell_list, 传入对应的输入通道数、隐藏通道数、卷积核大小等参数

        self.cell_list = nn.ModuleList(cell_list)
    #     使用nn.modulelist将所有单元整合为可训练模块

    def forward(self, input_tensor, hidden_state=None, pad_mask=None):
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
        input_tensor: 输入张量，形状为 (batch, seq, c, h, w) 或 (seq, batch, ...)。
        hidden_state: 隐藏状态的初始化值。
        pad_mask: 掩码，用于跳过无效时间步（可选）。
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor.permute(1, 0, 2, 3, 4)
        #     如果 batch_first=False，将输入张量的形状从 (seq, batch, ...) 调整为 (batch, seq, ...)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(
                batch_size=input_tensor.size(0), device=input_tensor.device
            )
        #     如果 hidden_state 被提供，则抛出 NotImplementedError（stateful 暂未实现）。
        # 否则，通过 _init_hidden 初始化隐藏状态。

        layer_output_list = []
        # 保存每一层的输出。
        last_state_list = []
        # 保存每一层的最后状态

        seq_len = input_tensor.size(1)
        # 输入序列的长度
        cur_layer_input = input_tensor
        # 当前层的输入张量（初始为输入张量本身）

        # 按层迭代
        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            # 从 hidden_state 中获取当前层的初始隐藏状态 (h, c)
            output_inner = []
            # 存储当前层每个时间步的输出
            for t in range(seq_len):
                # 对每个时间步t
                next_h, next_c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c]
                )
                if pad_mask is not None:
                    # True 表示 padding；padding 时间步不得改变该样本的记忆状态。
                    valid_t = (~pad_mask[:, t]).view(-1, 1, 1, 1)
                    h = torch.where(valid_t, next_h, h)
                    c = torch.where(valid_t, next_c, c)
                else:
                    h, c = next_h, next_c
                # 调用当前层的 ConvLSTMCell，传入对应时间步的数据和当前状态 (h, c)。
                output_inner.append(h)
            #     将隐藏状态 h 存入 output_inner

            layer_output = torch.stack(output_inner, dim=1)
            # 将时间步的输出堆叠为张量 layer_output
            cur_layer_input = layer_output
            # 将当前层的输出作为下一层的输入

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
        #     保存当前层的输出和最后状态。

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        #     如果 return_all_layers=False，仅保留最后一层的输出和状态。

        return layer_output_list, last_state_list
    # 返回最后一层的输出和最后状态。

    # 初始化每一层的隐藏状态 (h, c)，大小取决于输入张量的尺寸和每层的隐藏通道数。
    def _init_hidden(self, batch_size, device):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, device))
        return init_states

    # 检查 kernel_size 参数的类型是否合法
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

    # 如果参数为单值，扩展为与层数一致的列表
    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class DirectVSSConvLSTM(ConvLSTM):
    """ConvLSTM wrapper selecting the original direct four-gate VSS cell."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, cell_class=DirectVSSConvLSTMCell, **kwargs)


class VMambaV2ConvLSTM(ConvLSTM):
    """ConvLSTM wrapper selecting VMamba v2 for the candidate-state branch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, cell_class=VMambaV2ConvLSTMCell, **kwargs)


class Mamba2ConvLSTM(ConvLSTM):
    """ConvLSTM wrapper selecting Mamba2/SSD for the candidate-state branch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, cell_class=Mamba2ConvLSTMCell, **kwargs)


class ConvLSTM_Seg(nn.Module):
    def __init__(
        self, num_classes, input_size, input_dim, hidden_dim, kernel_size, pad_value=0
    ):
        super(ConvLSTM_Seg, self).__init__()
        self.convlstm_encoder = ConvLSTM(
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
    #     用于处理输入中的填充部分（如果输入长度不一）

    # input：输入张量，形状为 (batch_size, seq_len, input_dim, height, width)
    # batch_positions（可选）：时间序列的帧标记，未使用
    def forward(self, input, batch_positions=None):
        pad_mask = (
            (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
        )  # BxT pad mask
        # 生成填充掩码 pad_mask
        pad_mask = pad_mask if pad_mask.any() else None
        # 根据 pad_value 确定哪些时间步为填充帧。
        # 若没有填充帧（pad_mask.any() 为 False），pad_mask 设为 None
        _, states = self.convlstm_encoder(input, pad_mask=pad_mask)
        # 输入序列传入 ConvLSTM，获得隐藏状态和细胞状态
        # 提取最后一层的 最后一个时间步的细胞状态 states[0][1] 作为最终特征嵌入
        out = states[0][1]  # take last cell state as embedding
        out = self.classification_layer(out)
        # 使用卷积层将提取的嵌入映射为num_classes个类别的分割结果
        # 输出张量形状为(batch_size, num_classes, height, width)

        return out


class BConvLSTM_Seg(nn.Module):
    def __init__(
        self, num_classes, input_size, input_dim, hidden_dim, kernel_size, pad_value=0
    ):
        super(BConvLSTM_Seg, self).__init__()
        self.convlstm_forward = ConvLSTM(
            input_dim=input_dim,
            input_size=input_size,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            return_all_layers=False,
        )
        # 从时间序列的正向方向提取空间-时间特征。使用单向 ConvLSTM 的标准配置

        self.convlstm_backward = ConvLSTM(
            input_dim=input_dim,
            input_size=input_size,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            return_all_layers=False,
        )
        # 从时间序列的反向方向提取特征。通过翻转输入序列，将特征建模从最后一帧到第一帧。

        self.classification_layer = nn.Conv2d(
            in_channels=2 * hidden_dim,
            out_channels=num_classes,
            kernel_size=kernel_size,
            padding=1,
        )
        # 将正向和反向的隐藏状态特征进行拼接后，映射到 num_classes 个类别。
        self.pad_value = pad_value
    #     用于识别和处理输入中的填充帧（例如，序列长度不一时的补全）

    def forward(self, input, batch_posistions=None):
        pad_mask = (
            (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
        )  # BxT pad mask
        # 生成填充掩码 pad_mask，确定哪些时间步为填充帧（值全为 pad_value）若序列没有填充帧，pad_mask 设为 None
        pad_mask = pad_mask if pad_mask.any() else None

        # FORWARD
        # 输入序列 input 传入 convlstm_forward。
        # 获取最后一帧的细胞状态forward_states[0][1] 作为正向特征
        _, forward_states = self.convlstm_forward(input, pad_mask=pad_mask)
        out = forward_states[0][1]  # take last cell state as embedding

        # BACKWARD
        # 将输入序列在时间维度上翻转（使用 torch.flip）
        x_reverse = torch.flip(input, dims=[1])
        if pad_mask is not None:
            pmr = torch.flip(pad_mask.float(), dims=[1]).bool()
            x_reverse = torch.masked_fill(x_reverse, pmr[:, :, None, None, None], 0)
            # Fill leading padded positions with 0s
        #     若有填充帧，通过 pad_mask 将翻转后的填充部分设置为零。
        _, backward_states = self.convlstm_backward(x_reverse)
        # 输入翻转后的序列到 convlstm_backward。
        # 获取 最后一帧的细胞状态 backward_states[0][1] 作为反向特征

        out = torch.cat([out, backward_states[0][1]], dim=1)
        # 将正向特征和反向特征在通道维度上拼接
        out = self.classification_layer(out)
        # 将拼接后的特征通过卷积层，映射为 num_classes 的分割结果。 输出形状为 (batch_size, num_classes, height, width)
        return out


class BConvLSTM(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size):
        super(BConvLSTM, self).__init__()
        self.convlstm_forward = ConvLSTM(
            input_dim=input_dim,
            input_size=input_size,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            return_all_layers=False,
        )
        self.convlstm_backward = ConvLSTM(
            input_dim=input_dim,
            input_size=input_size,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            return_all_layers=False,
        )

    def forward(self, input, pad_mask=None):
        # FORWARD
        _, forward_states = self.convlstm_forward(input, pad_mask=pad_mask)
        out = forward_states[0][1]  # take last cell state as embedding

        # BACKWARD
        x_reverse = torch.flip(input, dims=[1])
        if pad_mask is not None:
            pmr = torch.flip(pad_mask.float(), dims=[1]).bool()
            x_reverse = torch.masked_fill(x_reverse, pmr[:, :, None, None, None], 0)
            # Fill leading padded positions with 0s
        _, backward_states = self.convlstm_backward(x_reverse)

        out = torch.cat([out, backward_states[0][1]], dim=1)
        return out
