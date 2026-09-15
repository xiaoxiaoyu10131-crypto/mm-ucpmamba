from pathlib import Path
import sys

import pytest
import torch
import torch.nn as nn


for _dependency in ("loguru", "einops", "timm"):
    pytest.importorskip(_dependency)

UTAE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UTAE_ROOT))

from src.backbones import convlstm
from src.backbones.mm_fusion import QualityAwareFusion2D
from src.backbones.mm_ucpmamba import MMUCPMamba
from src.backbones.vmamba import vmamba


class _TinyCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_proj = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.hidden_proj = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.vss = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.beta = nn.Parameter(torch.zeros(1))


class _TinyTemporalEncoder(nn.Module):
    """CPU wiring double; real Mamba1 requires the selective-scan CUDA backend."""

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, **kwargs):
        super().__init__()
        self.cell_list = nn.ModuleList([_TinyCell(input_dim, hidden_dim)])
        self.hidden_dim = hidden_dim
        self.height, self.width = input_size

    def forward(self, inputs, pad_mask=None):
        self.last_pad_mask = None if pad_mask is None else pad_mask.detach().clone()
        batch = inputs.shape[0]
        h = inputs.new_zeros(batch, self.hidden_dim, self.height, self.width)
        c = h.clone()
        outputs = []
        cell = self.cell_list[0]
        for t in range(inputs.shape[1]):
            candidate = torch.tanh(cell.input_proj(inputs[:, t]) + cell.hidden_proj(h))
            enhanced = cell.vss(candidate)
            candidate = candidate + cell.beta * (enhanced - candidate)
            next_c = 0.5 * c + candidate
            next_h = torch.tanh(next_c)
            if pad_mask is not None:
                valid = (~pad_mask[:, t]).view(-1, 1, 1, 1)
                h = torch.where(valid, next_h, h)
                c = torch.where(valid, next_c, c)
            else:
                h, c = next_h, next_c
            outputs.append(h)
        return [torch.stack(outputs, dim=1)], [[h, c]]


def _model(
    sar_orbits="both",
    fusion_mode="concat",
    use_quality_temporal=False,
    use_complementary_residual=True,
):
    return MMUCPMamba(
        optical_input_dim=10,
        sar_input_dim=3,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        out_conv=[32, 20],
        input_size=128,
        encoder_norm="group",
        hidden_dim=64,
        padding_mode="reflect",
        pad_value=0,
        temporal_encoder_class=_TinyTemporalEncoder,
        sar_orbits=sar_orbits,
        fusion_mode=fusion_mode,
        use_complementary_residual=use_complementary_residual,
        use_quality_temporal=use_quality_temporal,
    )


def _inputs():
    return (
        torch.randn(1, 5, 10, 128, 128),
        torch.randn(1, 7, 3, 128, 128),
        torch.randn(1, 9, 3, 128, 128),
    )


def test_mm_ucpmamba_three_modalities():
    model = _model().eval()
    s2, s1a, s1d = _inputs()
    with torch.no_grad():
        logits = model(s2, s1a, s1d)
    assert model.active_modalities == ("S2", "S1A", "S1D")
    assert logits.shape == (1, 20, 128, 128)
    assert torch.isfinite(logits).all()
    assert tuple(model.s2_branch.in_conv.conv.conv[0].weight.shape) == (64, 10, 3, 3)
    assert tuple(model.s1a_branch.in_conv.conv.conv[0].weight.shape) == (64, 3, 3, 3)
    assert tuple(model.s1d_branch.in_conv.conv.conv[0].weight.shape) == (64, 3, 3, 3)


def test_mm_ucpmamba_different_temporal_lengths():
    model = _model().eval()
    s2, s1a, s1d = _inputs()
    s2[:, -2:] = 0
    s1a[:, -1:] = 0
    with torch.no_grad():
        logits = model(
            s2, s1a, s1d,
            batch_positions_s2=torch.arange(5)[None],
            batch_positions_s1a=torch.arange(7)[None],
            batch_positions_s1d=torch.arange(9)[None],
        )
    assert logits.shape == (1, 20, 128, 128)
    assert torch.isfinite(logits).all()
    assert model.s2_branch.temporal_encoder.last_pad_mask.tolist() == [
        [False, False, False, True, True]
    ]
    assert model.s1a_branch.temporal_encoder.last_pad_mask.tolist() == [
        [False, False, False, False, False, False, True]
    ]
    assert not model.s1d_branch.temporal_encoder.last_pad_mask.any()


def test_mm_ucpmamba_independent_s1_branches():
    model = _model()
    branches = [model.s2_branch, model.s1a_branch, model.s1d_branch]
    assert len({id(branch) for branch in branches}) == 3
    parameter_storages = [{p.data_ptr() for p in branch.parameters()} for branch in branches]
    assert parameter_storages[0].isdisjoint(parameter_storages[1])
    assert parameter_storages[0].isdisjoint(parameter_storages[2])
    assert parameter_storages[1].isdisjoint(parameter_storages[2])
    betas = [branch.temporal_encoder.cell_list[0].beta for branch in branches]
    assert all(beta.item() == 0.0 for beta in betas)
    assert len({beta.data_ptr() for beta in betas}) == 3


def test_mm_ucpmamba_three_modal_fusion_shapes():
    model = _model().eval()
    fusion_inputs, fusion_outputs = [], []
    handles = []
    for fusion in model.fusions:
        handles.append(fusion.register_forward_pre_hook(
            lambda module, args: fusion_inputs.append((
                args[0].shape[0], sum(x.shape[1] for x in args), *args[0].shape[-2:]
            ))
        ))
        handles.append(fusion.register_forward_hook(
            lambda module, args, output: fusion_outputs.append(tuple(output.shape))
        ))
    with torch.no_grad():
        model(*_inputs())
    for handle in handles:
        handle.remove()
    assert fusion_inputs == [
        (1, 192, 128, 128), (1, 192, 64, 64),
        (1, 192, 32, 32), (1, 384, 16, 16),
    ]
    assert fusion_outputs == [
        (1, 64, 128, 128), (1, 64, 64, 64),
        (1, 64, 32, 32), (1, 128, 16, 16),
    ]


def test_mm_ucpmamba_modality_influence():
    torch.manual_seed(3)
    model = _model().eval()
    inputs = list(_inputs())
    with torch.no_grad():
        baseline = model(*inputs)
        for index in range(3):
            changed = [x.clone() for x in inputs]
            changed[index] = changed[index] + 0.25
            assert not torch.allclose(model(*changed), baseline)


def test_mm_ucpmamba_backward():
    torch.manual_seed(4)
    model = _model().train()
    logits = model(*_inputs())
    logits.square().mean().backward()
    for branch in (model.s2_branch, model.s1a_branch, model.s1d_branch):
        cell = branch.temporal_encoder.cell_list[0]
        assert branch.in_conv.conv.conv[0].weight.grad is not None
        assert cell.input_proj.weight.grad is not None
        assert cell.beta.grad is not None and torch.isfinite(cell.beta.grad).all()
        assert cell.vss.weight.grad is not None
        assert torch.count_nonzero(cell.vss.weight.grad) == 0
    assert model.fusions[0].proj.weight.grad is not None
    assert model.up_blocks[0].conv1.conv[0].weight.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

    model.zero_grad(set_to_none=True)
    for branch in (model.s2_branch, model.s1a_branch, model.s1d_branch):
        with torch.no_grad():
            branch.temporal_encoder.cell_list[0].beta.fill_(0.25)
    model(*_inputs()).square().mean().backward()
    for branch in (model.s2_branch, model.s1a_branch, model.s1d_branch):
        grad = branch.temporal_encoder.cell_list[0].vss.weight.grad
        assert grad is not None and torch.count_nonzero(grad) > 0


@pytest.mark.parametrize("sar_orbit", ["S1A", "S1D"])
def test_mm_ucpmamba_single_orbit_regression(sar_orbit):
    model = _model(sar_orbits=sar_orbit).eval()
    s2, s1a, s1d = _inputs()
    with torch.no_grad():
        logits = model(s2, s1a if sar_orbit == "S1A" else None,
                       s1d if sar_orbit == "S1D" else None)
    assert logits.shape == (1, 20, 128, 128)
    assert model.fusions[0].proj.in_channels == 128


def test_qaf_pixel_weights_initialization_and_edge_cases():
    fusion = QualityAwareFusion2D(8)
    optical = torch.randn(2, 8, 7, 7)
    sar = torch.randn(2, 8, 7, 7)
    fused, wo, ws = fusion(optical, sar, return_weights=True)
    assert torch.allclose(wo + ws, torch.ones_like(wo), atol=1e-6)
    assert fusion.gamma.item() == 0.0
    assert torch.isfinite(fused).all()

    model = _model(sar_orbits="S1A", fusion_mode="qaf", use_quality_temporal=True)
    s2, s1a, _ = _inputs()
    result = model(s2, s1a, return_debug=True)
    assert result["logits"].shape == (1, 20, 128, 128)
    result["logits"].square().mean().backward()
    assert model.s2_branch.temporal_encoder.cell_list[0].beta.grad is not None
    assert model.s1a_branch.temporal_encoder.cell_list[0].beta.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_qaf_can_disable_complementary_residual_for_e3():
    fusion = QualityAwareFusion2D(8, use_complementary_residual=False)
    optical = torch.randn(2, 8, 7, 7)
    sar = torch.randn(2, 8, 7, 7)
    fused, wo, ws = fusion(optical, sar, return_weights=True)
    expected = wo * fusion.optical_proj(optical) + ws * fusion.sar_proj(sar)
    assert torch.allclose(fused, expected, atol=1e-6)
    assert fusion.complementary is None and fusion.gamma is None
    fused.square().mean().backward()
    assert fusion.optical_proj.weight.grad is not None
    assert fusion.sar_proj.weight.grad is not None
    assert fusion.gate_logits.weight.grad is not None

    model = _model(
        sar_orbits="S1A", fusion_mode="qaf", use_complementary_residual=False
    )
    assert not any(
        "complementary" in name or name.endswith(".gamma")
        for name, _ in model.named_parameters()
    )


def test_mamba1_call_chain_is_used_by_default_cell():
    cell = convlstm.ConvLSTMCell((8, 8), 64, 64, (3, 3), True)
    assert cell.vmamba_mode == "g_only"
    assert cell.vss_variant == "vmamba1"
    assert isinstance(cell.vss[0], vmamba.VSSBlock)
    op = cell.vss[0].op
    assert isinstance(op, vmamba.SS2Dv2)
    assert op.forward_core.func.__name__ == "forward_corev2"
    assert op.forward_core.keywords["SelectiveScan"] is vmamba.SelectiveScanMamba
