"""独立空间分支在 1/4 尺度融合，深层编码与 CSM-ConvLSTM 共享。"""

import torch
import torch.nn as nn

from src.backbones.convlstm import ConvLSTM
from src.backbones.mm_fusion import LateConcatFusion2D
from src.backbones.mm_mid_components import (
    ShallowSpatialBranch,
    framewise_forward,
    masked_concat,
    masked_phenology,
    resize_valid,
)
from src.backbones.mm_temporal_alignment import DateWindowAligner, normalize_valid_mask
from src.backbones.utae import ConvBlock, DownConvBlock, PhenologyAttentionV2, UpConvBlock


class MMUCPMambaMid(nn.Module):
    """主路先融合后时序建模；浅层跳连仍逐模态聚合后融合。"""

    def __init__(
        self,
        optical_input_dim=10,
        sar_input_dim=3,
        encoder_widths=(64, 64, 64, 128),
        decoder_widths=(32, 32, 64, 128),
        out_conv=(32, 20),
        str_conv_k=4,
        str_conv_s=2,
        str_conv_p=1,
        input_size=128,
        encoder_norm="group",
        decoder_norm="batch",
        decoder_groups=8,
        hidden_dim=64,
        padding_mode="reflect",
        pad_value=0,
        temporal_encoder_class=ConvLSTM,
        sar_orbits="both",
        fusion_mode="concat",
        num_bins=24,
        start_day=None,
        end_day=None,
        modality_dropout=0.0,
        low_memory=False,
    ):
        super().__init__()
        if fusion_mode != "concat":
            raise ValueError("mm_ucpmamba_mid currently supports concat fusion only")
        if sar_orbits not in {"both", "S1A", "S1D"}:
            raise ValueError("sar_orbits must be 'both', 'S1A', or 'S1D'")
        if len(encoder_widths) != 4 or len(decoder_widths) != 4:
            raise ValueError("mid fusion expects four encoder and decoder stages")
        if decoder_widths[-1] != encoder_widths[-1]:
            raise ValueError("decoder bottleneck width must match encoder width")
        if decoder_norm not in {"batch", "group"} or decoder_groups <= 0:
            raise ValueError("invalid decoder normalization configuration")
        if not 0 <= modality_dropout <= 1:
            raise ValueError("modality_dropout must be in [0, 1]")
        self.aligner = DateWindowAligner(num_bins, start_day, end_day)
        self.sar_orbits = ("S1A", "S1D") if sar_orbits == "both" else (sar_orbits,)
        self.active_modalities = ("S2",) + self.sar_orbits
        self.encoder_widths = tuple(encoder_widths)
        self.input_size, self.pad_value = input_size, pad_value
        self.modality_dropout = float(modality_dropout)
        self.fusion_mode = fusion_mode
        self.input_channels = {"S2": optical_input_dim, "S1A": sar_input_dim, "S1D": sar_input_dim}
        sizes = [input_size]
        for _ in range(3):
            sizes.append((sizes[-1] + 2 * str_conv_p - str_conv_k) // str_conv_s + 1)
        if sizes[-1] <= 0 or any(
            (sizes[i + 1] - 1) * str_conv_s - 2 * str_conv_p + str_conv_k != sizes[i]
            for i in range(3)
        ):
            raise ValueError("input size and down/up convolution sizes are incompatible")
        self.sizes = tuple((size, size) for size in sizes)
        self.spatial_branches = nn.ModuleDict(
            {
                name: ShallowSpatialBranch(
                    self.input_channels[name],
                    encoder_widths[:3],
                    self.sizes[:3],
                    str_conv_k,
                    str_conv_s,
                    str_conv_p,
                    encoder_norm,
                    padding_mode,
                    low_memory=low_memory,
                )
                for name in self.active_modalities
            }
        )
        self.mid_fusion = LateConcatFusion2D(encoder_widths[2], len(self.active_modalities))
        self.skip_fusions = nn.ModuleList(
            LateConcatFusion2D(channels, len(self.active_modalities))
            for channels in encoder_widths[:2]
        )
        self.shared_down = DownConvBlock(
            encoder_widths[2],
            encoder_widths[3],
            str_conv_k,
            str_conv_s,
            str_conv_p,
            norm=encoder_norm,
            pad_value=None,
            padding_mode=padding_mode,
        )
        self.temporal_encoder = temporal_encoder_class(
            input_dim=encoder_widths[-1],
            input_size=self.sizes[-1],
            hidden_dim=hidden_dim,
            kernel_size=(3, 3),
        )
        self.out_convlstm = nn.Conv2d(hidden_dim, encoder_widths[-1], 3, padding=1)
        self.temporal_aggregator = PhenologyAttentionV2(encoder_widths[2], stage_level=2)
        self.up_blocks = nn.ModuleList(
            UpConvBlock(
                d_in=decoder_widths[i],
                d_out=decoder_widths[i - 1],
                d_skip=encoder_widths[i - 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                norm=decoder_norm,
                n_groups=decoder_groups,
                padding_mode=padding_mode,
            )
            for i in range(3, 0, -1)
        )
        self.out_conv = ConvBlock([decoder_widths[0]] + list(out_conv), padding_mode=padding_mode)

    @property
    def s2_branch(self):
        return self.spatial_branches["S2"]

    def _prepare_inputs(self, inputs, positions, pad_masks, valid_masks):
        reference = next(
            (inputs[n] for n in self.active_modalities if inputs[n] is not None), None
        )
        if reference is None:
            raise ValueError("at least one modality tensor is needed to infer batch shape")
        if reference.ndim != 5:
            raise ValueError("modality inputs must have shape [B,T,C,H,W]")
        b = reference.shape[0]
        valid, pads = {}, {}
        for name in self.active_modalities:
            x = inputs[name]
            if x is None:
                continue
            expected = (self.input_channels[name], self.input_size, self.input_size)
            if x.ndim != 5 or x.shape[0] != b or tuple(x.shape[2:]) != expected:
                raise ValueError(f"{name} input must have shape [B,T,{expected[0]},H,W]")
            if x.device != reference.device:
                raise ValueError("all modality tensors must be on the same device")
            pad = pad_masks.get(name)
            if pad is None:
                pad = (
                    x.eq(self.pad_value).flatten(2).all(dim=-1)
                    if self.pad_value is not None
                    else (torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device))
                )
            valid[name] = normalize_valid_mask(x, pad, valid_masks.get(name))
            pads[name] = pad
            if positions[name] is None:
                if valid[name].any():
                    raise ValueError(f"real observation dates are required for {name}")
                positions[name] = x.new_zeros(x.shape[:2])
            self.aligner.validate_dates(positions[name], valid[name])
        dropped = {}
        if self.training and self.modality_dropout > 0:
            names = list(valid)
            available = torch.stack([valid[n].flatten(1).any(dim=1) for n in names], dim=1)
            drop = torch.rand(b, len(names), device=reference.device) < self.modality_dropout
            lost = available.any(dim=1) & ~(available & ~drop).any(dim=1)
            drop[lost, available[lost].long().argmax(dim=1)] = False
            for i, name in enumerate(names):
                dropped[name] = drop[:, i]
                valid[name] = valid[name] & ~drop[:, i, None, None, None, None]
        return reference, valid, pads, dropped

    def forward(
        self,
        x_s2,
        x_s1a=None,
        x_s1d=None,
        batch_positions_s2=None,
        batch_positions_s1a=None,
        batch_positions_s1d=None,
        pad_masks=None,
        valid_masks=None,
        return_debug=False,
    ):
        inputs = {"S2": x_s2, "S1A": x_s1a, "S1D": x_s1d}
        positions = {
            "S2": batch_positions_s2,
            "S1A": batch_positions_s1a,
            "S1D": batch_positions_s1d,
        }
        reference, valid, pads, dropped = self._prepare_inputs(
            inputs, positions, pad_masks or {}, valid_masks or {}
        )
        b, k = reference.shape[0], self.aligner.num_bins
        aligned, aligned_valid, counts, shallow_skips, shallow_valid, shallow_pads = (
            {},
            {},
            {},
            {},
            {},
            {},
        )
        for name in self.active_modalities:
            if inputs[name] is None:
                aligned[name] = reference.new_zeros(b, k, self.encoder_widths[2], *self.sizes[2])
                aligned_valid[name] = torch.zeros(
                    b, k, 1, *self.sizes[2], dtype=torch.bool, device=reference.device
                )
                counts[name] = aligned_valid[name].long()
                shallow_skips[name] = [
                    reference.new_zeros(b, c, *self.sizes[i])
                    for i, c in enumerate(self.encoder_widths[:2])
                ]
                shallow_valid[name] = [
                    torch.zeros(b, 1, *self.sizes[i], dtype=torch.bool, device=reference.device)
                    for i in range(2)
                ]
                continue
            features, feature_valid, skips, skip_valid, frame_pad = self.spatial_branches[name](
                inputs[name], valid[name]
            )
            shallow_skips[name], shallow_valid[name], shallow_pads[name] = (
                skips,
                skip_valid,
                frame_pad,
            )
            aligned[name], aligned_valid[name], counts[name] = self.aligner(
                features,
                positions[name],
                pads[name],
                feature_valid,
            )
        ordered = self.active_modalities
        feature_dtype = next(
            (aligned[n].dtype for n in ordered if n in valid and valid[n].any()),
            reference.dtype,
        )
        aligned = {n: feature.to(feature_dtype) for n, feature in aligned.items()}
        shallow_skips = {
            n: [feature.to(feature_dtype) for feature in features]
            for n, features in shallow_skips.items()
        }
        flat_features = [aligned[n].reshape(b * k, -1, *self.sizes[2]) for n in ordered]
        flat_valid = [aligned_valid[n].reshape(b * k, 1, *self.sizes[2]) for n in ordered]
        fused = masked_concat(self.mid_fusion, flat_features, flat_valid).reshape(
            b, k, -1, *self.sizes[2]
        )
        pixel_valid = torch.stack([aligned_valid[n] for n in ordered]).any(dim=0)
        window_pad = ~pixel_valid.flatten(2).any(dim=-1)
        deep_valid = resize_valid(pixel_valid, self.sizes[3])
        temporal_pad = ~deep_valid.flatten(2).any(dim=-1)
        deep = framewise_forward(
            self.shared_down, fused, temporal_pad, self.encoder_widths[3], self.sizes[3]
        )
        deep = torch.where(deep_valid, deep, 0)
        _, states = self.temporal_encoder(deep, pad_mask=temporal_pad)
        bottleneck = self.out_convlstm(states[0][1])
        bottleneck = torch.where(deep_valid.any(dim=1), bottleneck, 0)
        skips = [
            masked_concat(
                fusion,
                [shallow_skips[n][i] for n in ordered],
                [shallow_valid[n][i] for n in ordered],
            )
            for i, fusion in enumerate(self.skip_fusions)
        ]
        skips.append(masked_phenology(self.temporal_aggregator, fused, pixel_valid))
        out = bottleneck
        for i, block in enumerate(self.up_blocks):
            out = block(out, skips[2 - i])
        logits = self.out_conv(out)
        if not return_debug:
            return logits
        return {
            "logits": logits,
            "aligned_valid": aligned_valid,
            "aligned_counts": counts,
            "window_pad_mask": window_pad,
            "temporal_pad_mask": temporal_pad,
            "pixel_valid": pixel_valid,
            "fused_sequence": fused,
            "fusion_sequence_shape": tuple(fused.shape),
            "temporal_input_shape": tuple(deep.shape),
            "shallow_pad_masks": shallow_pads,
            "dropped_modalities": dropped,
        }
