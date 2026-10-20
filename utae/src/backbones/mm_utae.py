"""Independent U-TAE branches with high-resolution late fusion for PASTIS-R."""

import math
from numbers import Real

import torch
from torch import nn

from src.backbones.utae import ConvBlock, UTAE


class MMUTAE(nn.Module):
    """Encode each modality with its own complete spatial/temporal U-TAE.

    The highest-resolution decoded features (32 channels by default) are
    concatenated in S2, S1A, S1D order and classified by a two-layer CNN.
    Temporal lengths, dates and padding masks remain independent; inputs are
    never temporally resampled. This is a local implementation of the paper's
    late-fusion design, not the authors' unpublished multimodal source code.

    The paper does not fully specify the fusion CNN's normalization/activation.
    ``legacy`` reuses this project's ConvBlock: Conv-BN-ReLU twice, including
    the final class layer, for controlled comparisons with MMUCPMamba.
    ``logits`` assumes Conv-BN-ReLU-Conv, with raw scores at the final layer.
    Optional auxiliary heads use the same convention and independent weights.
    Only a tail padded for every sample is trimmed independently per modality.
    The original LTAE time-axis GroupNorm remains: per-sample padding inside a
    batch can affect normalization, so cross-batch padding equivalence is not
    promised.
    """

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
        agg_mode="att_group",
        encoder_norm="group",
        n_head=16,
        d_model=256,
        d_k=4,
        pad_value=0,
        padding_mode="reflect",
        sar_orbits="both",
        auxiliary=False,
        head_mode="legacy",
    ):
        super().__init__()
        if sar_orbits not in {"both", "S1A", "S1D"}:
            raise ValueError("sar_orbits must be 'both', 'S1A', or 'S1D'")
        if head_mode not in {"legacy", "logits"}:
            raise ValueError("head_mode must be 'legacy' or 'logits'")
        if agg_mode not in {"att_group", "att_mean", "mean"}:
            raise ValueError("agg_mode must be 'att_group', 'att_mean', or 'mean'")
        if encoder_norm not in {"group", "batch", "instance"}:
            raise ValueError("encoder_norm must be 'group', 'batch', or 'instance'")
        if padding_mode not in {"zeros", "reflect", "replicate", "circular"}:
            raise ValueError("unsupported padding_mode")
        if not isinstance(pad_value, Real) or not math.isfinite(pad_value):
            raise ValueError("pad_value must be a finite number")
        encoder_widths = self._widths("encoder_widths", encoder_widths)
        decoder_widths = self._widths(
            "decoder_widths",
            encoder_widths if decoder_widths is None else decoder_widths,
        )
        out_conv = self._widths("out_conv", out_conv)
        if len(out_conv) != 2:
            raise ValueError("out_conv must contain exactly two layer widths")
        if len(encoder_widths) != len(decoder_widths):
            raise ValueError("encoder_widths and decoder_widths must have equal length")
        if encoder_widths[-1] != decoder_widths[-1]:
            raise ValueError("encoder and decoder bottleneck widths must match")
        for name, value in (
            ("optical_input_dim", optical_input_dim),
            ("sar_input_dim", sar_input_dim),
            ("n_head", n_head),
            ("d_model", d_model),
            ("d_k", d_k),
            ("str_conv_k", str_conv_k),
            ("str_conv_s", str_conv_s),
        ):
            self._positive_int(name, value)
        if (
            not isinstance(str_conv_p, int)
            or isinstance(str_conv_p, bool)
            or str_conv_p < 0
        ):
            raise ValueError("str_conv_p must be a non-negative integer")
        if encoder_widths[-1] % n_head or d_model % n_head:
            raise ValueError("bottleneck width and d_model must be divisible by n_head")
        if agg_mode == "att_group" and any(width % n_head for width in encoder_widths):
            raise ValueError(
                "att_group requires every encoder width divisible by n_head"
            )
        if encoder_norm == "group" and any(width % 4 for width in encoder_widths):
            raise ValueError(
                "group encoder normalization requires widths divisible by 4"
            )

        self.sar_orbits = ("S1A", "S1D") if sar_orbits == "both" else (sar_orbits,)
        self.active_modalities = ("S2",) + self.sar_orbits
        self.encoder_widths = encoder_widths
        self.decoder_widths = decoder_widths
        self.out_conv = out_conv
        self.pad_value = pad_value
        self.head_mode = head_mode
        self.auxiliary = bool(auxiliary)
        self.input_dims = {
            "S2": optical_input_dim,
            "S1A": sar_input_dim,
            "S1D": sar_input_dim,
        }
        self.str_conv_k, self.str_conv_s, self.str_conv_p = (
            str_conv_k,
            str_conv_s,
            str_conv_p,
        )
        self.padding_mode = padding_mode
        branch_kwargs = dict(
            encoder_widths=encoder_widths,
            decoder_widths=decoder_widths,
            out_conv=list(out_conv),
            str_conv_k=str_conv_k,
            str_conv_s=str_conv_s,
            str_conv_p=str_conv_p,
            agg_mode=agg_mode,
            encoder_norm=encoder_norm,
            n_head=n_head,
            d_model=d_model,
            d_k=d_k,
            encoder=True,
            pad_value=pad_value,
            padding_mode=padding_mode,
        )
        for modality in ("S2", "S1A", "S1D"):
            branch = None
            if modality in self.active_modalities:
                branch = UTAE(input_dim=self.input_dims[modality], **branch_kwargs)
                # Encoder mode bypasses this head; remove unused trainable parameters.
                branch.out_conv = nn.Identity()
            setattr(self, f"{modality.lower()}_branch", branch)
        feature_dim = decoder_widths[0]
        self.fusion_head = self._head(feature_dim * len(self.active_modalities))
        self.aux_heads = nn.ModuleDict(
            {modality: self._head(feature_dim) for modality in self.active_modalities}
            if self.auxiliary
            else {}
        )

    @staticmethod
    def _positive_int(name, value):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def _widths(cls, name, values):
        try:
            values = tuple(values)
        except TypeError as exc:
            raise ValueError(f"{name} must be a non-empty sequence of widths") from exc
        if not values:
            raise ValueError(f"{name} must be non-empty")
        for value in values:
            cls._positive_int(name, value)
        return values

    def _head(self, input_dim):
        hidden_dim, num_classes = self.out_conv
        if self.head_mode == "legacy":
            return ConvBlock(
                nkernels=[input_dim, hidden_dim, num_classes],
                padding_mode=self.padding_mode,
            )
        return nn.Sequential(
            nn.Conv2d(
                input_dim, hidden_dim, 3, padding=1, padding_mode=self.padding_mode
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(
                hidden_dim, num_classes, 3, padding=1, padding_mode=self.padding_mode
            ),
        )

    def _validate_input(self, modality, x, positions, reference):
        if not isinstance(x, torch.Tensor) or x.ndim != 5:
            raise ValueError(
                f"{modality} input must be a tensor with shape [B,T,C,H,W]"
            )
        if not x.is_floating_point():
            raise ValueError(f"{modality} input must have a floating-point dtype")
        b, t, c, h, w = x.shape
        if min(b, t, h, w) <= 0:
            raise ValueError(
                f"{modality} input must have non-empty batch, time and spatial axes"
            )
        if c != self.input_dims[modality]:
            raise ValueError(
                f"{modality} expects {self.input_dims[modality]} channels, got {c}"
            )
        if reference is not None:
            if (b, h, w) != (reference.shape[0], *reference.shape[-2:]):
                raise ValueError(
                    f"{modality} must match S2 batch and spatial dimensions"
                )
            if x.device != reference.device or x.dtype != reference.dtype:
                raise ValueError(f"{modality} must match S2 device and dtype")
        if not isinstance(positions, torch.Tensor) or positions.shape != (b, t):
            raise ValueError(f"{modality} dates must be a tensor with shape [B,T]")
        if positions.device != x.device:
            raise ValueError(
                f"{modality} dates must be on the same device as its input"
            )
        if (
            positions.dtype == torch.bool
            or positions.is_complex()
            or not torch.isfinite(positions).all()
        ):
            raise ValueError(f"{modality} dates must contain finite numeric values")
        pad_mask = (x == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
        if pad_mask.all(dim=1).any():
            raise ValueError(
                f"{modality} has a sample with an entirely padded sequence"
            )
        return pad_mask

    def _validate_spatial(self, x):
        sizes = [x.shape[-2:]]
        k, s, p = self.str_conv_k, self.str_conv_s, self.str_conv_p
        for _ in range(len(self.encoder_widths) - 1):
            previous = sizes[-1]
            current = tuple((size + 2 * p - k) // s + 1 for size in previous)
            if min(current) <= 0 or any(
                (size - 1) * s - 2 * p + k != old
                for size, old in zip(current, previous)
            ):
                raise ValueError(
                    "spatial size is incompatible with UTAE down/up convolutions"
                )
            sizes.append(current)
        if self.padding_mode == "reflect" and any(min(size) <= 1 for size in sizes):
            raise ValueError(
                "reflect padding requires every spatial stage to exceed one pixel"
            )

    def forward(
        self,
        x_s2,
        x_s1a=None,
        x_s1d=None,
        batch_positions_s2=None,
        batch_positions_s1a=None,
        batch_positions_s1d=None,
        return_aux=False,
    ):
        """Return fused scores, or a dict with scores and enabled auxiliary scores.

        Every enabled modality requires its own [B,T] dates and at least one
        non-padding frame per sample. Timestamp zero is valid and never treated
        as padding. Auxiliary outputs exist only when ``auxiliary=True``.
        """
        inputs = {"S2": x_s2, "S1A": x_s1a, "S1D": x_s1d}
        positions = {
            "S2": batch_positions_s2,
            "S1A": batch_positions_s1a,
            "S1D": batch_positions_s1d,
        }
        pad_masks = {}
        for modality in self.active_modalities:
            pad_masks[modality] = self._validate_input(
                modality,
                inputs[modality],
                positions[modality],
                x_s2 if modality != "S2" else None,
            )
        self._validate_spatial(x_s2)
        features = {}
        for modality in self.active_modalities:
            branch = getattr(self, f"{modality.lower()}_branch")
            valid_times = (~pad_masks[modality]).any(dim=0).nonzero(as_tuple=False)
            length = int(valid_times[-1, 0].item()) + 1
            sequence = inputs[modality][:, :length].contiguous()
            dates = positions[modality][:, :length].to(sequence.dtype)
            feature, _ = branch(sequence, batch_positions=dates)
            features[modality] = feature
        logits = self.fusion_head(torch.cat(list(features.values()), dim=1))
        if not return_aux:
            return logits
        outputs = {"logits": logits}
        if self.auxiliary:
            for modality, feature in features.items():
                outputs[f"logits_{modality.lower()}"] = self.aux_heads[modality](
                    feature
                )
        return outputs
