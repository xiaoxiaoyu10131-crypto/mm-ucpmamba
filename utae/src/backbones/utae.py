"""
U-TAE Implementation
Author: Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.backbones.convlstm import ConvLSTM, BConvLSTM
from src.backbones.ltae import LTAE2d

class UTAE(nn.Module):
    def __init__(
        self,
        input_dim,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        out_conv=[32, 20],
        str_conv_k=4,
        str_conv_s=2,
        str_conv_p=1,
        agg_mode="att_group",
        encoder_norm="group",
        n_head=16,
        d_model=256,
        d_k=4,
        encoder=False,
        return_maps=False,
        pad_value=0,
        padding_mode="reflect",
    ):
        """
        U-TAE architecture for spatio-temporal encoding of satellite image time series.
        Args:
            input_dim (int): Number of channels in the input images.
            encoder_widths (List[int]): List giving the number of channels of the successive encoder_widths of the convolutional encoder.
            This argument also defines the number of encoder_widths (i.e. the number of downsampling steps +1)
            in the architecture.
            The number of channels are given from top to bottom, i.e. from the highest to the lowest resolution.
            decoder_widths (List[int], optional): Same as encoder_widths but for the decoder. The order in which the number of
            channels should be given is also from top to bottom. If this argument is not specified the decoder
            will have the same configuration as the encoder.
            out_conv (List[int]): Number of channels of the successive convolutions for the
            str_conv_k (int): Kernel size of the strided up and down convolutions.
            str_conv_s (int): Stride of the strided up and down convolutions.
            str_conv_p (int): Padding of the strided up and down convolutions.
            agg_mode (str): Aggregation mode for the skip connections. Can either be:
                - att_group (default) : Attention weighted temporal average, using the same
                channel grouping strategy as in the LTAE. The attention masks are bilinearly
                resampled to the resolution of the skipped feature maps.
                - att_mean : Attention weighted temporal average,
                 using the average attention scores across heads for each date.
                - mean : Temporal average excluding padded dates.
            encoder_norm (str): Type of normalisation layer to use in the encoding branch. Can either be:
                - group : GroupNorm (default)
                - batch : BatchNorm
                - instance : InstanceNorm
            n_head (int): Number of heads in LTAE.
            d_model (int): Parameter of LTAE
            d_k (int): Key-Query space dimension
            encoder (bool): If true, the feature maps instead of the class scores are returned (default False)
            return_maps (bool): If true, the feature maps instead of the class scores are returned (default False)
            pad_value (float): Value used by the dataloader for temporal padding.
            padding_mode (str): Spatial padding strategy for convolutional layers (passed to nn.Conv2d).
        """
        super(UTAE, self).__init__()
        self.n_stages = len(encoder_widths)
        self.return_maps = return_maps
        self.encoder_widths = encoder_widths
        self.decoder_widths = decoder_widths
        self.enc_dim = (
            decoder_widths[0] if decoder_widths is not None else encoder_widths[0]
        )
        self.stack_dim = (
            sum(decoder_widths) if decoder_widths is not None else sum(encoder_widths)
        )
        self.pad_value = pad_value
        self.encoder = encoder
        if encoder:
            self.return_maps = True

        if decoder_widths is not None:
            assert len(encoder_widths) == len(decoder_widths)
            assert encoder_widths[-1] == decoder_widths[-1]
        else:
            decoder_widths = encoder_widths

        self.in_conv = ConvBlock(
            nkernels=[input_dim] + [encoder_widths[0], encoder_widths[0]],
            pad_value=pad_value,
            norm=encoder_norm,
            padding_mode=padding_mode,
        )
        self.down_blocks = nn.ModuleList(
            DownConvBlock(
                d_in=encoder_widths[i],
                d_out=encoder_widths[i + 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                pad_value=pad_value,
                norm=encoder_norm,
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1)
        )
        self.up_blocks = nn.ModuleList(
            UpConvBlock(
                d_in=decoder_widths[i],
                d_out=decoder_widths[i - 1],
                d_skip=encoder_widths[i - 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                norm="batch",
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1, 0, -1)
        )
        self.temporal_encoder = LTAE2d(
            in_channels=encoder_widths[-1],
            d_model=d_model,
            n_head=n_head,
            mlp=[d_model, encoder_widths[-1]],
            return_att=True,
            d_k=d_k,
        )
        self.temporal_aggregator = Temporal_Aggregator(mode=agg_mode)
        self.out_conv = ConvBlock(nkernels=[decoder_widths[0]] + out_conv, padding_mode=padding_mode)

    def forward(self, input, batch_positions=None, return_att=False):
        pad_mask = (
            (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
        )  # BxT pad mask
        out = self.in_conv.smart_forward(input)
        feature_maps = [out]
        # SPATIAL ENCODER
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i].smart_forward(feature_maps[-1])
            feature_maps.append(out)
        # TEMPORAL ENCODER
        out, att = self.temporal_encoder(
            feature_maps[-1], batch_positions=batch_positions, pad_mask=pad_mask
        )
        # SPATIAL DECODER
        if self.return_maps:
            maps = [out]
        for i in range(self.n_stages - 1):
            skip = self.temporal_aggregator(
                feature_maps[-(i + 2)], pad_mask=pad_mask, attn_mask=att
            )
            out = self.up_blocks[i](out, skip)
            if self.return_maps:
                maps.append(out)

        if self.encoder:
            return out, maps
        else:
            out = self.out_conv(out)
            if return_att:
                return out, att
            if self.return_maps:
                return out, maps
            else:
                return out


class TemporallySharedBlock(nn.Module):
    """
    Helper module for convolutional encoding blocks that are shared across a sequence.
    This module adds the self.smart_forward() method the the block.
    smart_forward will combine the batch and temporal dimension of an input tensor
    if it is 5-D and apply the shared convolutions to all the (batch x temp) positions.
    """

    def __init__(self, pad_value=None):
        super(TemporallySharedBlock, self).__init__()
        self.out_shape = None
        self.pad_value = pad_value

    def smart_forward(self, input):
        if len(input.shape) == 4:
            return self.forward(input)
        else:
            b, t, c, h, w = input.shape

            if self.pad_value is not None:
                dummy = torch.zeros(input.shape, device=input.device).float()
                self.out_shape = self.forward(dummy.view(b * t, c, h, w)).shape

            out = input.view(b * t, c, h, w)
            if self.pad_value is not None:
                pad_mask = (out == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
                if pad_mask.any():
                    temp = (
                        torch.ones(
                            self.out_shape, device=input.device, requires_grad=False
                        )
                        * self.pad_value
                    )
                    temp[~pad_mask] = self.forward(out[~pad_mask])
                    out = temp
                else:
                    out = self.forward(out)
            else:
                out = self.forward(out)
            _, c, h, w = out.shape
            out = out.view(b, t, c, h, w)
            return out


class ConvLayer(nn.Module):
    def __init__(
        self,
        nkernels,
        norm="batch",
        k=3,
        s=1,
        p=1,
        n_groups=4,
        last_relu=True,
        last_norm=True,
        padding_mode="reflect",
    ):
        super(ConvLayer, self).__init__()
        layers = []
        if norm == "batch":
            nl = nn.BatchNorm2d
        elif norm == "instance":
            nl = nn.InstanceNorm2d
        elif norm == "group":
            nl = lambda num_feats: nn.GroupNorm(
                num_channels=num_feats,
                num_groups=n_groups,
            )
        else:
            nl = None
        for i in range(len(nkernels) - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=nkernels[i],
                    out_channels=nkernels[i + 1],
                    kernel_size=k,
                    padding=p,
                    stride=s,
                    padding_mode=padding_mode,
                )
            )
            if nl is not None:
                layers.append(nl(nkernels[i + 1]))

            if last_relu:
                layers.append(nn.ReLU())
            elif i < len(nkernels) - 2:
                layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, input):
        return self.conv(input)


class ConvBlock(TemporallySharedBlock):
    def __init__(
        self,
        nkernels,
        pad_value=None,
        norm="batch",
        last_relu=True,
        padding_mode="reflect",
    ):
        super(ConvBlock, self).__init__(pad_value=pad_value)
        self.conv = ConvLayer(
            nkernels=nkernels,
            norm=norm,
            last_relu=last_relu,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        return self.conv(input)


class DownConvBlock(TemporallySharedBlock):
    def __init__(
        self,
        d_in,
        d_out,
        k,
        s,
        p,
        pad_value=None,
        norm="batch",
        padding_mode="reflect",
    ):
        super(DownConvBlock, self).__init__(pad_value=pad_value)
        self.down = ConvLayer(
            nkernels=[d_in, d_in],
            norm=norm,
            k=k,
            s=s,
            p=p,
            padding_mode=padding_mode,
        )
        self.conv1 = ConvLayer(
            nkernels=[d_in, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        out = self.down(input)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out


class UpConvBlock(nn.Module):
    def __init__(
        self, d_in, d_out, k, s, p, norm="batch", d_skip=None,
        padding_mode="reflect", n_groups=4
    ):
        super(UpConvBlock, self).__init__()
        d = d_out if d_skip is None else d_skip
        if norm == "group":
            if n_groups <= 0 or d % n_groups != 0 or d_out % n_groups != 0:
                raise ValueError(
                    "GroupNorm groups must be positive and divide decoder channels"
                )

            def decoder_norm(channels):
                return nn.GroupNorm(num_groups=n_groups, num_channels=channels)
        else:
            decoder_norm = nn.BatchNorm2d
        self.skip_conv = nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=d, kernel_size=1),
            decoder_norm(d),
            nn.ReLU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=d_in, out_channels=d_out, kernel_size=k, stride=s, padding=p
            ),
            decoder_norm(d_out),
            nn.ReLU(),
        )
        self.conv1 = ConvLayer(
            nkernels=[d_out + d, d_out],
            norm=norm,
            n_groups=n_groups,
            padding_mode=padding_mode,
        )
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out],
            norm=norm,
            n_groups=n_groups,
            padding_mode=padding_mode,
        )

    def forward(self, input, skip):
        out = self.up(input)
        out = torch.cat([out, self.skip_conv(skip)], dim=1)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out


class Temporal_Aggregator(nn.Module):
    def __init__(self, mode="mean"):
        super(Temporal_Aggregator, self).__init__()
        self.mode = mode

    def forward(self, x, pad_mask=None, attn_mask=None):
        if pad_mask is not None and pad_mask.any():
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w)

                if x.shape[-2] > w:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)

                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                attn = attn * (~pad_mask).float()[None, :, :, None, None]

                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(
                    size=x.shape[-2:], mode="bilinear", align_corners=False
                )(attn)
                attn = attn * (~pad_mask).float()[:, :, None, None]
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                out = x * (~pad_mask).float()[:, :, None, None, None]
                out = out.sum(dim=1) / (~pad_mask).sum(dim=1)[:, None, None, None]
                return out
        else:
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w)
                if x.shape[-2] > w:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)
                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(
                    size=x.shape[-2:], mode="bilinear", align_corners=False
                )(attn)
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                return x.mean(dim=1)


class PhenologyMultiScaleAttention(nn.Module):
    def __init__(self, in_channels, stage_level, scales=(3,5), out_channels=64):
        super().__init__()
        self.stage = stage_level
        self.scales = scales
        self.out_channels = out_channels
        
        # 根据不同尺度动态生成多分支 conv3d
        # 每个分支 kernel_size = (k,3,3)，padding = (k//2,1,1)
        self.convs = nn.ModuleList([
            nn.Conv3d(in_channels, out_channels, 
                      kernel_size=(k,3,3), 
                      padding=(k//2,1,1))
            for k in scales
        ])
        
        # 融合各分支后再压缩到单通道做注意力
        self.fusion_conv = nn.Conv3d(len(scales)*out_channels, out_channels, kernel_size=1)
        self.attn_net = nn.Sequential(
            nn.ReLU(),
            # 直接对 H×W 池化
            nn.AdaptiveAvgPool3d((None, 1, 1)),
            nn.Conv3d(out_channels, 1, kernel_size=1)
        )
        
        # 残差融合权重
        self.alpha = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, x, pad_mask):
        # x: [B, T, C, H, W]  -> 变成 [B, C, T, H, W]
        x3d = x.permute(0,2,1,3,4)
        
        # 多尺度分支
        feats = [conv(x3d) for conv in self.convs]  
        # feats[i]: [B, out_channels, T, H, W]
        
        # 拼接 & 融合
        multi = torch.cat(feats, dim=1)                    # [B, out*len(scales), T, H, W]
        multi = self.fusion_conv(multi)                    # [B, out_channels, T, H, W]
        
        # 计算注意力得分
        attn = self.attn_net(multi)                        # [B,1,T,1,1]
        attn = attn.masked_fill(pad_mask.view(x.size(0), 1, -1, 1, 1), -1e9)
        attn_w = F.softmax(attn, dim=2)                    # 沿时间维度归一
        
        # 加权求和 & masked mean
        weighted = (x * attn_w.permute(0,2,1,3,4)).sum(dim=1)  # [B,C,H,W]
        mask = (~pad_mask)[:, :, None, None, None].float()
        mean_valid = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        
        alpha = torch.sigmoid(self.alpha)  # 约束在(0,1)
        return alpha * weighted + (1-alpha) * mean_valid


class SharedSkipAttention(nn.Module):
    def __init__(self, shared_channels=64, max_stage_idx=2):
        """
        使用一个共享 attention 模块处理前 max_stage_idx 层的跳跃连接
        """
        super().__init__()
        self.shared_attn = PhenologyMultiScaleAttention(
            in_channels=shared_channels,
            stage_level=0  # 仅用于结构一致，实际不使用
        )
        self.max_stage_idx = max_stage_idx

    def forward(self, feature, pad_mask, stage_idx):
        """
        feature: Tensor of shape [B, T, C, H, W]
        pad_mask: BoolTensor of shape [B, T]
        stage_idx: 当前处理的跳跃层索引（0-based）

        Returns:
            [B, C, H, W]：注意力聚合结果（如果stage_idx > max_stage_idx，则返回mean）
        """
        if stage_idx > self.max_stage_idx:
            # 跳过不处理的高层，返回 mean（不使用 attention）
            mask = (~pad_mask)[:, :, None, None, None].float()
            return (feature * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        
        return self.shared_attn(feature, pad_mask)



class PhenologyAttentionV2(nn.Module):
    def __init__(self, in_channels, stage_level):
        super().__init__()
        self.stage = stage_level
        
        # out_channels = 24 if stage_level in [0, 1] else 32
        out_channels = 64
        # 层级自适应参数
        if stage_level in [0, 1]:  # 浅层（高分辨率）
            self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=(3,3,3), padding=(1,1,1))
        else:  # 深层（低分辨率）
            self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=(3,1,1), padding=(1,0,0))
        
        self.attn_net = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((None, 1, 1)),
            nn.Conv3d(out_channels, 1, kernel_size=1)
        )
        
        # 残差权重（可学习）
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 初始权重偏向均值
        # self.dropout = nn.Dropout3d(p=0.1)

    def forward(self, x, pad_mask):
        B, T, C, H, W = x.shape

        # -------------------------- 分支1：注意力加权（带掩码）--------------------------
        # 1. 时空特征提取（适配3D卷积维度）
        x_3d = self.conv3d(x.permute(0, 2, 1, 3, 4))  # (B,T,C,H,W) → (B,C,T,H,W)
        # 2. 生成原始注意力得分：ReLU → AdaptiveAvgPool3d → Conv3d 1×1×1
        attn = self.attn_net(x_3d)  # (B,64,T,H,W) → (B,1,T,1,1)
        # 3. 【掩码处理】：无效时间步设为-1e9（显式在Conv3d 1×1×1和Softmax之间）
        attn = attn.masked_fill(pad_mask.view(B, 1, T, 1, 1), -1e9)
        # 4. Softmax归一化，得到注意力权重
        attn_weights = F.softmax(attn, dim=2)  # (B,1,T,1,1)
        # 5. 注意力加权求和
        weighted_sum = (x * attn_weights.permute(0, 2, 1, 3, 4)).sum(dim=1)  # (B,C,H,W)

        # -------------------------- 分支2：时间平均（显式加掩码）--------------------------
        # 1. 【掩码处理】：显式放在Mean前面，过滤无效时间步
        # 生成和x同维度的掩码：(B,T) → (B,T,1,1,1)，有效时间步=1，无效=0
        mean_mask = (~pad_mask)[:, :, None, None, None].float()  # 有效日期为 1，padding 日期为 0
        # 2. 用掩码过滤无效时间步（无效时间步特征置0）
        x_valid = x * mean_mask  # (B,T,C,H,W) → 无效时间步特征被清零
        # 3. 时间平均：只对有效时间步求和，再除以有效时间步数量（避免除以0）
        sum_valid = x_valid.sum(dim=1)  # 有效时间步特征求和 → (B,C,H,W)
        count_valid = mean_mask.sum(dim=1).clamp(min=1.0)  # 有效时间步数量 → (B,1,1,1)
        mean_sum = sum_valid / count_valid  # 显式掩码后的时间平均

        # -------------------------- 残差融合 --------------------------
        alpha = torch.sigmoid(self.alpha)  # 约束权重在(0,1)
        return alpha * weighted_sum + (1 - alpha) * mean_sum


class RecUNet(nn.Module):
    """Recurrent U-Net architecture. Similar to the U-TAE architecture but
    the L-TAE is replaced by a recurrent network
    and temporal averages are computed for the skip connections."""

    def __init__(
        self,
        input_dim,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        out_conv=[32, 20],
        str_conv_k=4,
        str_conv_s=2,
        str_conv_p=1,
        temporal="lstm",
        input_size=128,
        encoder_norm="group",
        hidden_dim=128,
        encoder=False,
        padding_mode="reflect",
        pad_value=0,
        temporal_encoder_class=ConvLSTM,
        skip_aggregation="phenology",
        decoder_norm="batch",
    ):
        super(RecUNet, self).__init__()
        self.n_stages = len(encoder_widths)
        self.temporal = temporal
        self.encoder_widths = encoder_widths
        self.decoder_widths = decoder_widths
        self.enc_dim = (
            decoder_widths[0] if decoder_widths is not None else encoder_widths[0]
        )
        self.stack_dim = (
            sum(decoder_widths) if decoder_widths is not None else sum(encoder_widths)
        )
        self.pad_value = pad_value

        self.encoder = encoder
        if encoder:
            self.return_maps = True
        else:
            self.return_maps = False

        if decoder_widths is not None:
            assert len(encoder_widths) == len(decoder_widths)
            assert encoder_widths[-1] == decoder_widths[-1]
        else:
            decoder_widths = encoder_widths

        self.in_conv = ConvBlock(
            nkernels=[input_dim] + [encoder_widths[0], encoder_widths[0]],
            pad_value=pad_value,
            norm=encoder_norm,
            padding_mode=padding_mode,
        )

        self.down_blocks = nn.ModuleList(
            DownConvBlock(
                d_in=encoder_widths[i],
                d_out=encoder_widths[i + 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                pad_value=pad_value,
                norm=encoder_norm,
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1)
        )
        self.up_blocks = nn.ModuleList(
            UpConvBlock(
                d_in=decoder_widths[i],
                d_out=decoder_widths[i - 1],
                d_skip=encoder_widths[i - 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                norm=decoder_norm,
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1, 0, -1)
        )
        # self.temporal_aggregator = Temporal_Aggregator(mode="mean")
        # self.temporal_aggregator = nn.ModuleList([PhenologyAttention() for _ in range(self.n_stages - 1)])
        # self.temporal_aggregator = nn.ModuleList()
        # for i in range(self.n_stages - 1):  # 只处理前三层
        #     self.temporal_aggregator.append(
        #          PhenologyMultiScaleAttention(in_channels=self.encoder_widths[i], stage_level = i)
        #      )
        # self.temporal_aggregator = ContentBasedAttention(hidden_dim=1024)
        # self.attention_blocks = nn.ModuleList([ContentBasedAttention(hidden_dim=128) for _ in range(self.n_stages - 1)])
        # self.attention_blocks = nn.ModuleList([Eca(128) for _ in range(self.n_stages - 1)])
        # self.temporal_aggregator = SharedSkipAttention(shared_channels=64) #原来是这一行未注释
        # -------------------------- 关键修改1：初始化PhenologyAttentionV2模块列表 --------------------------
        self.skip_aggregation = skip_aggregation
        if skip_aggregation == "phenology":
            self.temporal_aggregator = nn.ModuleList(
                PhenologyAttentionV2(
                    in_channels=encoder_widths[i],
                    stage_level=i,
                )
                for i in range(self.n_stages - 1)
            )
        elif skip_aggregation == "phenology_deep_mean":
            # Keep phenology attention only on the deepest skip (stage n-2).
            # The shallower skips use the same masked temporal mean as UConvLSTM.
            self.temporal_aggregator = nn.ModuleList(
                PhenologyAttentionV2(
                    in_channels=encoder_widths[i],
                    stage_level=i,
                )
                if i == self.n_stages - 2
                else Temporal_Aggregator(mode="mean")
                for i in range(self.n_stages - 1)
            )
        elif skip_aggregation == "mean":
            self.temporal_aggregator = Temporal_Aggregator(mode="mean")
        else:
            raise ValueError(
                "skip_aggregation must be 'phenology', 'phenology_deep_mean', or 'mean'"
            )
        # --------------------------------------------------------------------------------------------------

        if temporal == "mean":
            self.temporal_encoder = Temporal_Aggregator(mode="mean")
        elif temporal == "lstm":
            size = int(input_size / str_conv_s ** (self.n_stages - 1))
            self.temporal_encoder = temporal_encoder_class(
                input_dim=encoder_widths[-1],
                input_size=(size, size),
                hidden_dim=hidden_dim,
                kernel_size=(3, 3),
            )
            self.out_convlstm = nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=encoder_widths[-1],
                kernel_size=3,
                padding=1,
            )
        elif temporal == "blstm":
            size = int(input_size / str_conv_s ** (self.n_stages - 1))
            self.temporal_encoder = BConvLSTM(
                input_dim=encoder_widths[-1],
                input_size=(size, size),
                hidden_dim=hidden_dim,
                kernel_size=(3, 3),
            )
            self.out_convlstm = nn.Conv2d(
                in_channels=2 * hidden_dim,
                out_channels=encoder_widths[-1],
                kernel_size=3,
                padding=1,
            )
        elif temporal == "mono":
            self.temporal_encoder = None
        self.out_conv = ConvBlock(nkernels=[decoder_widths[0]] + out_conv, padding_mode=padding_mode)
        # self.ssncj = Posi_attn(channel=20)
        # self.sat = BasicLayer(num_layers=1, n_iter=1, stoken_size=[2, 2], num_heads=4)
        # self.shsa = SHSA(dim=None, qk_dim=1)


    def forward(self, input, batch_positions=None):
        pad_mask = (
            (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
        )  # BxT pad mask

        out = self.in_conv.smart_forward(input)

        feature_maps = [out]
        # ENCODER
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i].smart_forward(feature_maps[-1])
            feature_maps.append(out)

        # Temporal encoder
        if self.temporal == "mean":
            out = self.temporal_encoder(feature_maps[-1], pad_mask=pad_mask)
        elif self.temporal == "lstm":
            _, out = self.temporal_encoder(feature_maps[-1], pad_mask=pad_mask)
            out = out[0][1]  # take last cell state as embedding
            out = self.out_convlstm(out)
        elif self.temporal == "blstm":
            out = self.temporal_encoder(feature_maps[-1], pad_mask=pad_mask)
            out = self.out_convlstm(out)
        elif self.temporal == "mono":
            out = feature_maps[-1]

        if self.return_maps:
            maps = [out]
        for i in range(self.n_stages - 1):
            if self.temporal != "mono":
                if self.skip_aggregation in ("phenology", "phenology_deep_mean"):
                    encoder_stage = self.n_stages - i - 2
                    skip = self.temporal_aggregator[encoder_stage](
                        feature_maps[-(i + 2)], pad_mask
                    )
                else:
                    skip = self.temporal_aggregator(
                        feature_maps[-(i + 2)], pad_mask=pad_mask
                    )
                #skip = self.temporal_aggregator(feature_maps[-(i + 2)], pad_mask, stage_idx = i)  #原来是这一行没注释，上一行注释
            else:
                skip = feature_maps[-(i + 2)]
            out = self.up_blocks[i](out, skip)
            if self.return_maps:
                maps.append(out)

        if self.encoder:
            return out, maps
        else:
            # out = self.sat(out)
            # out = self.shsa(out)
            out = self.out_conv(out)
            # out = self.shsa(out)
            # out = self.sat(out)
            if self.return_maps:
                return out, maps
            else:
                return out
