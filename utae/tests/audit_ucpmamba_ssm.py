"""Print reproducible implementation facts for the three UCPMamba SSM routes."""

from pathlib import Path
from types import SimpleNamespace
import json
import sys

import torch


UTAE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UTAE_ROOT))

from src import model_utils
from src.backbones.vmamba.mamba2 import ss2d as mamba2_ss2d
from src.backbones.vmamba.mamba2 import ssd_minimal


def config(model):
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


def describe(route):
    model = model_utils.get_model(config(route))
    cell = model.temporal_encoder.cell_list[0]
    op = cell.vss[0].op
    scan_class = op.forward_core.keywords.get("SelectiveScan")
    return {
        "route": route,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "ssm_dispatch_class": type(op).__name__,
        "ssm_implementation": next(
            base.__name__
            for base in type(op).__mro__
            if base.__name__ in {"SS2Dv2", "SS2Dm0"}
        ),
        "forward_core": op.forward_core.func.__name__,
        "scan": (
            scan_class.__name__
            if scan_class is not None
            else "selective_scan_chunk_fn"
        ),
        "state_dimension": getattr(op, "d_state", op.A_logs.shape[-1]),
        "vss_variant": cell.vss_variant,
    }


def trace_mamba2():
    model = model_utils.get_model(config("ucpmamba_mamba2"))
    core = model.temporal_encoder.cell_list[0].vss[0].op
    calls = ["SS2Dm0.forward"]
    original_core = core.forward_core
    original_scan = mamba2_ss2d.selective_scan_chunk_fn
    original_chunk = ssd_minimal.mamba_chunk_scan_combined_torch
    original_ssd = ssd_minimal.ssd_minimal_discrete

    def traced_core(*args, **kwargs):
        calls.append("SS2Dm0.forward_corem0")
        return original_core(*args, **kwargs)

    def traced_scan(*args, **kwargs):
        calls.append("selective_scan_chunk_fn")
        kwargs["backend"] = "torch"
        return original_scan(*args, **kwargs)

    def traced_chunk(*args, **kwargs):
        calls.append("mamba_chunk_scan_combined_torch")
        return original_chunk(*args, **kwargs)

    def traced_ssd(*args, **kwargs):
        calls.append("ssd_minimal_discrete")
        return original_ssd(*args, **kwargs)

    core.forward_core = traced_core
    mamba2_ss2d.selective_scan_chunk_fn = traced_scan
    ssd_minimal.mamba_chunk_scan_combined_torch = traced_chunk
    ssd_minimal.ssd_minimal_discrete = traced_ssd
    try:
        x = torch.randn(1, 64, 8, 8, requires_grad=True)
        y = core(x)
        y.square().mean().backward()
    finally:
        core.forward_core = original_core
        mamba2_ss2d.selective_scan_chunk_fn = original_scan
        ssd_minimal.mamba_chunk_scan_combined_torch = original_chunk
        ssd_minimal.ssd_minimal_discrete = original_ssd
    return {
        "runtime_call_stack": calls,
        "input_shape": list(x.shape),
        "output_shape": list(y.shape),
        "input_gradient_finite": bool(torch.isfinite(x.grad).all()),
    }


if __name__ == "__main__":
    routes = ["ucpmamba", "ucpmamba_vmambav2", "ucpmamba_mamba2"]
    result = {
        "routes": [describe(route) for route in routes],
        "mamba2_runtime_trace": trace_mamba2(),
    }
    print(json.dumps(result, indent=2))
