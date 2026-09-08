from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
import torch.nn as nn


UTAE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UTAE_ROOT))

from src import model_utils
from src.backbones import convlstm
from src.backbones.vmamba.mamba2 import SS2Dm0
from src.backbones.vmamba import vmamba
from src.backbones.vmamba.mamba2 import ss2d as mamba2_ss2d
from src.backbones.vmamba.mamba2 import ssd_minimal


def _config(model):
    return SimpleNamespace(
        model=model,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        out_conv=[32, 20],
        str_conv_k=4,
        str_conv_s=2,
        str_conv_p=1,
        encoder_norm="group",
        padding_mode="reflect",
        pad_value=0,
    )


def test_cells_change_only_the_vss_core_configuration(monkeypatch):
    calls = []

    class FakeMSVSS(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            calls.append(kwargs)

        def forward(self, x):
            return x

    monkeypatch.setattr(convlstm, "MSVSS", FakeMSVSS)
    cells = [
        convlstm.ConvLSTMCell((8, 8), 4, 8, (3, 3), True),
        convlstm.VMambaV2ConvLSTMCell((8, 8), 4, 8, (3, 3), True),
        convlstm.Mamba2ConvLSTMCell((8, 8), 4, 8, (3, 3), True),
    ]

    assert calls[0] == {"dim": 8, "drop_path": 0.01}
    assert calls[1] == {
        "dim": 8,
        "drop_path": 0.01,
        **convlstm.VMAMBA_V2_SSM_KWARGS,
    }
    assert calls[2] == {
        "dim": 8,
        "drop_path": 0.01,
        **convlstm.MAMBA2_SSM_KWARGS,
    }
    assert [cell.vss_variant for cell in cells] == [
        "vmamba1",
        "vmambav2",
        "mamba2",
    ]
    assert all(cell.vmamba_mode == "g_only" for cell in cells)
    assert all(torch.equal(cells[0].beta, cell.beta) for cell in cells[1:])


def test_model_routes_preserve_all_non_temporal_configuration(monkeypatch):
    monkeypatch.setattr(model_utils.ucpmamba, "UCPMamba", lambda **kwargs: kwargs)
    expected_classes = {
        "ucpmamba": convlstm.ConvLSTM,
        "ucpmamba_vmambav2": convlstm.VMambaV2ConvLSTM,
        "ucpmamba_mamba2": convlstm.Mamba2ConvLSTM,
    }
    configs = {}
    for route, expected_class in expected_classes.items():
        kwargs = model_utils.get_model(_config(route))
        assert kwargs.pop("temporal_encoder_class") is expected_class
        assert kwargs["skip_aggregation"] == "phenology"
        configs[route] = kwargs
    assert configs["ucpmamba"] == configs["ucpmamba_vmambav2"]
    assert configs["ucpmamba"] == configs["ucpmamba_mamba2"]


def test_routes_resolve_to_distinct_ssm_implementations():
    original = convlstm.ConvLSTMCell((8, 8), 4, 64, (3, 3), True)
    vmambav2 = convlstm.VMambaV2ConvLSTMCell((8, 8), 4, 64, (3, 3), True)
    mamba2 = convlstm.Mamba2ConvLSTMCell((8, 8), 4, 64, (3, 3), True)

    original_op = original.vss[0].op
    vmambav2_op = vmambav2.vss[0].op
    mamba2_op = mamba2.vss[0].op
    assert isinstance(original_op, vmamba.SS2Dv2)
    assert isinstance(vmambav2_op, vmamba.SS2Dv2)
    assert isinstance(mamba2_op, SS2Dm0)
    assert original_op.forward_core.func.__name__ == "forward_corev2"
    assert vmambav2_op.forward_core.func.__name__ == "forward_corev2"
    assert mamba2_op.forward_core.func.__name__ == "forward_corem0"
    assert original_op.forward_core.keywords["SelectiveScan"] is vmamba.SelectiveScanMamba
    assert vmambav2_op.forward_core.keywords["SelectiveScan"] is vmamba.SelectiveScanOflex
    assert vmambav2_op.forward_core.keywords["no_einsum"] is True


def test_model_parameter_shapes_outside_vss_are_unchanged():
    states = {
        route: model_utils.get_model(_config(route)).state_dict()
        for route in ("ucpmamba", "ucpmamba_vmambav2", "ucpmamba_mamba2")
    }
    prefix = "temporal_encoder.cell_list.0.vss."
    original_outside_vss = {
        name: tensor.shape
        for name, tensor in states["ucpmamba"].items()
        if not name.startswith(prefix)
    }
    for route in ("ucpmamba_vmambav2", "ucpmamba_mamba2"):
        outside_vss = {
            name: tensor.shape
            for name, tensor in states[route].items()
            if not name.startswith(prefix)
        }
        assert outside_vss == original_outside_vss


def test_mamba2_executes_chunked_ssd_and_backpropagates(monkeypatch):
    calls = []
    original_scan = mamba2_ss2d.selective_scan_chunk_fn
    original_chunk = ssd_minimal.mamba_chunk_scan_combined_torch
    original_ssd = ssd_minimal.ssd_minimal_discrete

    def traced_ssd(*args, **kwargs):
        calls.append("ssd_minimal_discrete")
        return original_ssd(*args, **kwargs)

    def traced_scan(*args, **kwargs):
        calls.append("selective_scan_chunk_fn")
        kwargs["backend"] = "torch"
        return original_scan(*args, **kwargs)

    def traced_chunk(*args, **kwargs):
        calls.append("mamba_chunk_scan_combined_torch")
        return original_chunk(*args, **kwargs)

    monkeypatch.setattr(ssd_minimal, "ssd_minimal_discrete", traced_ssd)
    monkeypatch.setattr(ssd_minimal, "mamba_chunk_scan_combined_torch", traced_chunk)
    monkeypatch.setattr(mamba2_ss2d, "selective_scan_chunk_fn", traced_scan)
    core = SS2Dm0(d_model=64, d_state=64, ssm_ratio=1.0, channel_first=True)
    x = torch.randn(1, 64, 8, 8, requires_grad=True)
    y = core(x)
    y.square().mean().backward()

    assert y.shape == x.shape
    assert calls == [
        "selective_scan_chunk_fn",
        "mamba_chunk_scan_combined_torch",
        "ssd_minimal_discrete",
    ]
    assert x.grad is not None and torch.isfinite(x.grad).all()
    core_grads = [p.grad for p in core.parameters() if p.requires_grad]
    assert all(grad is not None and torch.isfinite(grad).all() for grad in core_grads)
    assert any(torch.count_nonzero(grad) for grad in core_grads)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_full_model_cuda_forward_backward_matches_original_shape():
    pytest.importorskip("selective_scan_cuda_oflex")
    inputs = torch.randn(1, 1, 10, 128, 128, device="cuda")
    with torch.no_grad():
        original = model_utils.get_model(_config("ucpmamba")).cuda().eval()
        original_shape = original(inputs).shape
    del original

    model = model_utils.get_model(_config("ucpmamba_mamba2")).cuda().train()
    outputs = model(inputs)
    outputs.square().mean().backward()
    assert outputs.shape == original_shape == (1, 20, 128, 128)
    assert torch.isfinite(outputs).all()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
    assert model.temporal_encoder.cell_list[0].beta.grad is not None
