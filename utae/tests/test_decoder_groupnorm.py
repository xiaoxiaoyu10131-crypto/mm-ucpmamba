import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn


def _load_utae_without_cuda_backends(monkeypatch):
    utae_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(utae_root))

    class _Placeholder(nn.Module):
        pass

    convlstm = types.ModuleType("src.backbones.convlstm")
    convlstm.ConvLSTM = _Placeholder
    convlstm.BConvLSTM = _Placeholder
    monkeypatch.setitem(sys.modules, "src.backbones.convlstm", convlstm)

    ltae = types.ModuleType("src.backbones.ltae")
    ltae.LTAE2d = _Placeholder
    monkeypatch.setitem(sys.modules, "src.backbones.ltae", ltae)

    module_path = utae_root / "src" / "backbones" / "utae.py"
    spec = importlib.util.spec_from_file_location("_decoder_norm_utae", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_upconv_groupnorm_replaces_all_internal_batchnorm(monkeypatch):
    utae = _load_utae_without_cuda_backends(monkeypatch)
    block = utae.UpConvBlock(
        d_in=64,
        d_out=32,
        d_skip=64,
        k=4,
        s=2,
        p=1,
        norm="group",
        n_groups=8,
    )

    group_norms = [m for m in block.modules() if isinstance(m, nn.GroupNorm)]
    batch_norms = [m for m in block.modules() if isinstance(m, nn.BatchNorm2d)]
    assert len(group_norms) == 4
    assert not batch_norms
    assert all(m.num_groups == 8 for m in group_norms)

    x = torch.randn(2, 64, 8, 8, requires_grad=True)
    skip = torch.randn(2, 64, 16, 16, requires_grad=True)
    output = block(x, skip)
    assert output.shape == (2, 32, 16, 16)
    output.mean().backward()
    assert x.grad is not None
    assert skip.grad is not None


def test_upconv_batchnorm_default_is_unchanged(monkeypatch):
    utae = _load_utae_without_cuda_backends(monkeypatch)
    block = utae.UpConvBlock(
        d_in=64,
        d_out=32,
        d_skip=64,
        k=4,
        s=2,
        p=1,
    )

    assert len([m for m in block.modules() if isinstance(m, nn.BatchNorm2d)]) == 4
    assert not [m for m in block.modules() if isinstance(m, nn.GroupNorm)]


def test_upconv_groupnorm_rejects_non_divisible_channels(monkeypatch):
    utae = _load_utae_without_cuda_backends(monkeypatch)
    with pytest.raises(ValueError, match="divide decoder channels"):
        utae.UpConvBlock(
            d_in=64,
            d_out=30,
            d_skip=64,
            k=4,
            s=2,
            p=1,
            norm="group",
            n_groups=8,
        )


class _TinyTemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, **kwargs):
        super().__init__()
        self.projection = nn.Conv2d(input_dim, hidden_dim, 1)

    def forward(self, inputs, pad_mask=None):
        hidden = self.projection(inputs.mean(dim=1))
        return [inputs], [[hidden, hidden]]


def _load_mm_ucpmamba_without_cuda_backends(monkeypatch):
    utae = _load_utae_without_cuda_backends(monkeypatch)
    monkeypatch.setitem(sys.modules, "src.backbones.utae", utae)

    utae_root = Path(__file__).resolve().parents[1]
    module_path = utae_root / "src" / "backbones" / "mm_ucpmamba.py"
    spec = importlib.util.spec_from_file_location(
        "src.backbones.mm_ucpmamba", module_path
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "src.backbones.mm_ucpmamba", module)
    spec.loader.exec_module(module)
    return module


def test_mm_ucpmamba_groupnorm_scope_and_backward(monkeypatch):
    mm_ucpmamba = _load_mm_ucpmamba_without_cuda_backends(monkeypatch)
    model = mm_ucpmamba.MMUCPMamba(
        optical_input_dim=10,
        sar_input_dim=3,
        encoder_widths=[8, 8, 8, 16],
        decoder_widths=[8, 8, 8, 16],
        out_conv=[8, 20],
        input_size=16,
        encoder_norm="group",
        decoder_norm="group",
        decoder_groups=8,
        hidden_dim=8,
        temporal_encoder_class=_TinyTemporalEncoder,
        sar_orbits="both",
        fusion_mode="concat",
    )

    decoder_batch_norms = [
        module
        for block in model.up_blocks
        for module in block.modules()
        if isinstance(module, nn.BatchNorm2d)
    ]
    decoder_group_norms = [
        module
        for block in model.up_blocks
        for module in block.modules()
        if isinstance(module, nn.GroupNorm)
    ]
    output_batch_norms = [
        module for module in model.out_conv.modules()
        if isinstance(module, nn.BatchNorm2d)
    ]

    assert not decoder_batch_norms
    assert len(decoder_group_norms) == 12
    assert all(module.num_groups == 8 for module in decoder_group_norms)
    assert len(output_batch_norms) == 2
    assert output_batch_norms[-1].num_features == 20

    inputs = (
        torch.randn(2, 2, 10, 16, 16),
        torch.randn(2, 2, 3, 16, 16),
        torch.randn(2, 2, 3, 16, 16),
    )
    output = model(*inputs)
    assert output.shape == (2, 20, 16, 16)
    output.mean().backward()
    assert model.up_blocks[0].up[0].weight.grad is not None
