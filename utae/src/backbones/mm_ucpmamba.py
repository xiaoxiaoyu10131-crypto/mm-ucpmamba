"""Independent Optical-SAR UCPMamba with multi-scale late fusion."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from src.backbones.convlstm import ConvLSTM
from src.backbones.mm_fusion import (
    FeatureQualityEstimator,
    LateConcatFusion2D,
    QualityAwareFusion2D,
    QualityGuidedPhenologyAttentionV2,
)
from src.backbones.utae import ConvBlock, DownConvBlock, PhenologyAttentionV2, UpConvBlock


def _masked_mean(sequence, pad_mask):
    valid = (~pad_mask)[:, :, None, None, None].to(dtype=sequence.dtype)
    return (sequence * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


@dataclass
class BranchFeatures:
    skip0: torch.Tensor
    skip1: torch.Tensor
    skip2: torch.Tensor
    bottleneck: torch.Tensor
    q0: Optional[torch.Tensor] = None
    q1: Optional[torch.Tensor] = None
    q2: Optional[torch.Tensor] = None
    qb: Optional[torch.Tensor] = None
    temporal_attention: Optional[dict] = None


class UCPMambaTemporalBranch(nn.Module):
    """A standalone UCPMamba spatial encoder and temporal bottleneck branch."""

    def __init__(
        self,
        input_dim,
        encoder_widths=(64, 64, 64, 128),
        str_conv_k=4,
        str_conv_s=2,
        str_conv_p=1,
        encoder_norm="group",
        hidden_dim=64,
        input_size=128,
        padding_mode="reflect",
        pad_value=0,
        temporal_encoder_class=ConvLSTM,
        use_quality_temporal=False,
        needs_quality=False,
        quality_lambda=1.0,
    ):
        super().__init__()
        self.encoder_widths = list(encoder_widths)
        self.n_stages = len(self.encoder_widths)
        self.pad_value = pad_value
        self.use_quality_temporal = use_quality_temporal
        self.needs_quality = needs_quality or use_quality_temporal

        self.in_conv = ConvBlock(
            nkernels=[input_dim, self.encoder_widths[0], self.encoder_widths[0]],
            pad_value=pad_value,
            norm=encoder_norm,
            padding_mode=padding_mode,
        )
        self.down_blocks = nn.ModuleList(
            DownConvBlock(
                d_in=self.encoder_widths[i],
                d_out=self.encoder_widths[i + 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                pad_value=pad_value,
                norm=encoder_norm,
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1)
        )

        size = int(input_size / str_conv_s ** (self.n_stages - 1))
        self.temporal_encoder = temporal_encoder_class(
            input_dim=self.encoder_widths[-1],
            input_size=(size, size),
            hidden_dim=hidden_dim,
            kernel_size=(3, 3),
        )
        self.out_convlstm = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=self.encoder_widths[-1],
            kernel_size=3,
            padding=1,
        )

        if use_quality_temporal:
            aggregator_class = QualityGuidedPhenologyAttentionV2
        else:
            aggregator_class = PhenologyAttentionV2
        self.temporal_aggregator = nn.ModuleList(
            aggregator_class(
                in_channels=self.encoder_widths[i],
                stage_level=i,
                **({"lambda_q": quality_lambda} if use_quality_temporal else {}),
            )
            for i in range(self.n_stages - 1)
        )
        self.quality_estimators = (
            nn.ModuleList(FeatureQualityEstimator(self.encoder_widths[i]) for i in range(self.n_stages - 1))
            if self.needs_quality
            else None
        )
        self.bottleneck_quality = (
            FeatureQualityEstimator(self.encoder_widths[-1]) if self.needs_quality else None
        )

    def forward(self, x, positions=None, quality=None, return_debug=False):
        if x.ndim != 5:
            raise ValueError("branch input must have shape [B,T,C,H,W]")
        pad_mask = (x == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
        out = self.in_conv.smart_forward(x)
        feature_maps = [out]
        for block in self.down_blocks:
            feature_maps.append(block.smart_forward(feature_maps[-1]))

        _, states = self.temporal_encoder(feature_maps[-1], pad_mask=pad_mask)
        out = self.out_convlstm(states[0][1])

        skips = [None] * (self.n_stages - 1)
        qualities = [None] * (self.n_stages - 1)
        attentions = {}
        for stage in range(self.n_stages - 1):
            sequence = feature_maps[stage]
            q_sequence = None
            if self.needs_quality:
                q_sequence = self.quality_estimators[stage](sequence, quality)
                qualities[stage] = _masked_mean(q_sequence, pad_mask)
            aggregator = self.temporal_aggregator[stage]
            if self.use_quality_temporal:
                skip, attn = aggregator(
                    sequence, pad_mask, quality=q_sequence, return_attention=True
                )
                attentions[stage] = attn
            else:
                skip = aggregator(sequence, pad_mask)
            skips[stage] = skip

        qb = self.bottleneck_quality(out) if self.needs_quality else None
        result = BranchFeatures(
            skip0=skips[0],
            skip1=skips[1],
            skip2=skips[2],
            bottleneck=out,
            q0=qualities[0],
            q1=qualities[1],
            q2=qualities[2],
            qb=qb,
            temporal_attention=attentions or None,
        )
        return result


class MMUCPMamba(nn.Module):
    """Independent S2/S1-orbit UCPMamba branches and one shared decoder."""

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
        hidden_dim=64,
        padding_mode="reflect",
        pad_value=0,
        temporal_encoder_class=ConvLSTM,
        sar_orbits="both",
        fusion_mode="concat",
        use_complementary_residual=True,
        use_quality_temporal=False,
        quality_lambda=1.0,
        modality_dropout=0.0,
    ):
        super().__init__()
        if fusion_mode not in {"concat", "qaf"}:
            raise ValueError("fusion_mode must be 'concat' or 'qaf'")
        if sar_orbits not in {"both", "S1A", "S1D"}:
            raise ValueError("sar_orbits must be 'both', 'S1A', or 'S1D'")
        if sar_orbits == "both" and fusion_mode != "concat":
            raise ValueError("three-modality baseline supports concat fusion only")
        if len(encoder_widths) != len(decoder_widths) or len(encoder_widths) != 4:
            raise ValueError("MMUCPMamba expects four encoder and decoder stages")
        if not 0.0 <= modality_dropout <= 1.0:
            raise ValueError("modality_dropout must be in [0, 1]")

        self.fusion_mode = fusion_mode
        self.sar_orbits = ("S1A", "S1D") if sar_orbits == "both" else (sar_orbits,)
        self.active_modalities = ("S2",) + self.sar_orbits
        self.use_complementary_residual = bool(use_complementary_residual)
        self.modality_dropout = float(modality_dropout)
        self.encoder_widths = list(encoder_widths)
        self.decoder_widths = list(decoder_widths)
        needs_quality = fusion_mode == "qaf" or use_quality_temporal
        branch_kwargs = dict(
            encoder_widths=encoder_widths,
            str_conv_k=str_conv_k,
            str_conv_s=str_conv_s,
            str_conv_p=str_conv_p,
            encoder_norm=encoder_norm,
            hidden_dim=hidden_dim,
            input_size=input_size,
            padding_mode=padding_mode,
            pad_value=pad_value,
            temporal_encoder_class=temporal_encoder_class,
            use_quality_temporal=use_quality_temporal,
            needs_quality=needs_quality,
            quality_lambda=quality_lambda,
        )
        self.s2_branch = UCPMambaTemporalBranch(input_dim=optical_input_dim, **branch_kwargs)
        self.s1a_branch = (
            UCPMambaTemporalBranch(input_dim=sar_input_dim, **branch_kwargs)
            if "S1A" in self.sar_orbits else None
        )
        self.s1d_branch = (
            UCPMambaTemporalBranch(input_dim=sar_input_dim, **branch_kwargs)
            if "S1D" in self.sar_orbits else None
        )

        if fusion_mode == "qaf":
            self.fusions = nn.ModuleList(
                QualityAwareFusion2D(
                    channels,
                    use_complementary_residual=self.use_complementary_residual,
                )
                for channels in encoder_widths
            )
        else:
            self.fusions = nn.ModuleList(
                LateConcatFusion2D(channels, len(self.active_modalities))
                for channels in encoder_widths
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
            for i in range(len(encoder_widths) - 1, 0, -1)
        )
        self.out_conv = ConvBlock(
            nkernels=[decoder_widths[0]] + list(out_conv), padding_mode=padding_mode
        )

    def reset_quality_fusion_initialization(self):
        """Restore zero-init QAF gates after the project's generic weight_init."""
        if self.fusion_mode != "qaf":
            return
        for fusion in self.fusions:
            nn.init.zeros_(fusion.gate_logits.weight)
            nn.init.zeros_(fusion.gate_logits.bias)
            if fusion.gamma is not None:
                nn.init.zeros_(fusion.gamma)

    @property
    def opt_branch(self):
        """Backward-compatible read-only name for the S2 branch."""
        return self.s2_branch

    @property
    def sar_branch(self):
        """Backward-compatible name, valid only for a single-orbit model."""
        if len(self.sar_orbits) != 1:
            raise AttributeError("tri-modal models have separate s1a_branch and s1d_branch")
        return self.s1a_branch if self.s1a_branch is not None else self.s1d_branch

    @staticmethod
    def _drop_modalities(branches, probability):
        dropped = {name: False for name in branches}
        if probability <= 0.0:
            return branches, dropped
        first = next(iter(branches.values()))
        draws = torch.rand(
            first.bottleneck.shape[0],
            len(branches),
            device=first.bottleneck.device,
        ) < probability
        draws[draws.all(dim=1), 0] = False

        def mask_branch(branch, drop):
            keep = (~drop).to(dtype=branch.bottleneck.dtype).view(-1, 1, 1, 1)
            for name in ("skip0", "skip1", "skip2", "bottleneck"):
                setattr(branch, name, getattr(branch, name) * keep)

        for index, (name, branch) in enumerate(branches.items()):
            mask_branch(branch, draws[:, index])
            dropped[name] = bool(draws[:, index].any().item())
        return branches, dropped

    def decode(self, bottleneck, skips):
        out = bottleneck
        for i, block in enumerate(self.up_blocks):
            out = block(out, skips[len(skips) - i - 1])
        return self.out_conv(out)

    def forward(
        self,
        x_s2,
        x_s1a=None,
        x_s1d=None,
        batch_positions_s2=None,
        batch_positions_s1a=None,
        batch_positions_s1d=None,
        quality_s2=None,
        quality_s1a=None,
        quality_s1d=None,
        return_aux=False,
        return_debug=False,
    ):
        inputs = {"S2": x_s2, "S1A": x_s1a, "S1D": x_s1d}
        positions = {
            "S2": batch_positions_s2,
            "S1A": batch_positions_s1a,
            "S1D": batch_positions_s1d,
        }
        qualities = {"S2": quality_s2, "S1A": quality_s1a, "S1D": quality_s1d}
        branch_modules = {
            "S2": self.s2_branch,
            "S1A": self.s1a_branch,
            "S1D": self.s1d_branch,
        }
        branches = {}
        for name in self.active_modalities:
            if inputs[name] is None:
                raise ValueError(
                    f"{name} input is required for active modalities {self.active_modalities}"
                )
            branches[name] = branch_modules[name](
                inputs[name],
                positions=positions[name],
                quality=qualities[name],
                return_debug=return_debug,
            )
        branches, dropped = self._drop_modalities(
            branches, self.modality_dropout if self.training else 0.0
        )
        features = {
            name: [branch.skip0, branch.skip1, branch.skip2, branch.bottleneck]
            for name, branch in branches.items()
        }
        fused_features = []
        fusion_debug = []
        for i, fusion in enumerate(self.fusions):
            if self.fusion_mode == "qaf":
                sar_name = self.sar_orbits[0]
                q_s2 = [
                    branches["S2"].q0, branches["S2"].q1,
                    branches["S2"].q2, branches["S2"].qb,
                ][i]
                q_sar = [
                    branches[sar_name].q0, branches[sar_name].q1,
                    branches[sar_name].q2, branches[sar_name].qb,
                ][i]
                fused, wo, ws = fusion(
                    features["S2"][i], features[sar_name][i], q_s2, q_sar,
                    return_weights=True,
                )
                fusion_debug.append({"wo": wo, "ws": ws})
            else:
                fused = fusion(*(features[name][i] for name in self.active_modalities))
            fused_features.append(fused)

        logits = self.decode(fused_features[3], fused_features[:3])
        if not return_aux and not return_debug:
            return logits

        result = {"logits": logits}
        if return_aux:
            for name, branch in branches.items():
                result[f"logits_{name.lower()}"] = self.decode(
                    branch.bottleneck, features[name][:3]
                )
        if return_debug:
            result["fusion_weights"] = fusion_debug
            for name, branch in branches.items():
                key = name.lower()
                result[f"quality_{key}"] = [
                    branch.q0, branch.q1, branch.q2, branch.qb
                ]
                result[f"temporal_attention_{key}"] = branch.temporal_attention
                result[f"beta_{key}"] = (
                    branch_modules[name].temporal_encoder.cell_list[0].beta
                )
                result[f"dropped_{key}"] = dropped[name]
        return result
