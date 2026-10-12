"""Real spatial/fusion/skip wiring with a CPU-only temporal test double."""

import ast
import importlib
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]


class CpuTemporalDouble(nn.Module):
    """Checks state skipping and gradients; does not validate Mamba or CUDA."""

    def __init__(self, input_dim, hidden_dim, input_size, **kwargs):
        super().__init__()
        self.projection = nn.Conv2d(input_dim, hidden_dim, 1)
        self.hidden_dim, self.input_size = hidden_dim, input_size

    def forward(self, inputs, pad_mask=None):
        self.last_inputs = inputs.detach().clone()
        self.last_pad_mask = pad_mask.detach().clone()
        state = inputs.new_zeros(inputs.shape[0], self.hidden_dim, *self.input_size)
        outputs = []
        for index in range(inputs.shape[1]):
            candidate = torch.tanh(self.projection(inputs[:, index]) + 0.5 * state)
            keep = (~pad_mask[:, index])[:, None, None, None]
            state = torch.where(keep, candidate, state)
            outputs.append(state)
        return [torch.stack(outputs, dim=1)], [[state, state]]


@pytest.fixture
def mid_module(monkeypatch):
    """Restore all src modules after import, including modules imported indirectly."""
    def is_src(key):
        return key == "src" or key.startswith("src.")

    original = {key: value for key, value in sys.modules.items() if is_src(key)}
    for key in original:
        sys.modules.pop(key)
    source = types.ModuleType("src")
    source.__path__ = [str(ROOT / "src")]
    backbones = types.ModuleType("src.backbones")
    backbones.__path__ = [str(ROOT / "src/backbones")]
    source.backbones = backbones
    sys.modules.update({"src": source, "src.backbones": backbones})
    convlstm = types.ModuleType("src.backbones.convlstm")
    convlstm.ConvLSTM = CpuTemporalDouble
    convlstm.BConvLSTM = CpuTemporalDouble
    ltae = types.ModuleType("src.backbones.ltae")
    ltae.LTAE2d = nn.Module
    sys.modules.update({"src.backbones.convlstm": convlstm, "src.backbones.ltae": ltae})
    monkeypatch.syspath_prepend(str(ROOT))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield importlib.import_module("src.backbones.mm_ucpmamba_mid")
    finally:
        for key in list(sys.modules):
            if is_src(key):
                sys.modules.pop(key)
        sys.modules.update(original)
        torch.set_num_threads(previous_threads)


def model(module, **kwargs):
    options = dict(
        optical_input_dim=10, sar_input_dim=3,
        encoder_widths=[8, 8, 8, 16], decoder_widths=[8, 8, 8, 16],
        out_conv=[8, 4], input_size=16, hidden_dim=8,
        decoder_norm="group", decoder_groups=8,
        temporal_encoder_class=CpuTemporalDouble,
        num_bins=4, start_day=0, end_day=40,
    )
    options.update(kwargs)
    return module.MMUCPMambaMid(**options)


def inputs():
    torch.manual_seed(13)
    return [torch.randn(2, length, channels, 16, 16)
            for length, channels in [(3, 10), (4, 3), (2, 3)]]


def dates():
    return dict(
        batch_positions_s2=torch.tensor([[1., 9., 25.], [2., 7., 21.]]),
        batch_positions_s1a=torch.tensor([[3., 12., 18., 28.], [4., 11., 19., 22.]]),
        batch_positions_s1d=torch.tensor([[13., 27.], [15., 29.]]),
    )


def temporal(model_instance):
    modules = [item for item in model_instance.modules() if isinstance(item, CpuTemporalDouble)]
    assert len(modules) == 1
    return modules[0]


def test_non_equal_lengths_use_dates_and_shared_temporal(mid_module):
    instance = model(mid_module).eval()
    with torch.no_grad():
        debug = instance(*inputs(), **dates(), return_debug=True)
    assert debug["logits"].shape == (2, 4, 16, 16)
    assert torch.isfinite(debug["logits"]).all()
    assert debug["window_pad_mask"].tolist() == [[False, False, False, True]] * 2
    expected = {"S2": [2, 0, 1, 0], "S1A": [1, 2, 1, 0], "S1D": [0, 1, 1, 0]}
    for name, counts in expected.items():
        assert debug["aligned_counts"][name][0, :, 0, 0, 0].tolist() == counts
    assert tuple(debug["fusion_sequence_shape"]) == (2, 4, 8, 4, 4)
    assert tuple(debug["temporal_input_shape"]) == (2, 4, 16, 2, 2)
    shared = temporal(instance)
    assert torch.equal(shared.last_pad_mask, debug["temporal_pad_mask"])
    assert shared.last_inputs[:, -1].count_nonzero() == 0


def test_missing_modality_and_all_empty_windows_block_bias(mid_module):
    instance = model(mid_module).eval()
    for layer in instance.modules():
        if isinstance(layer, (nn.Conv2d, nn.Conv3d)) and layer.bias is not None:
            nn.init.constant_(layer.bias, 0.7)
    x = inputs()
    pad = {key: torch.ones(2, value.shape[1], dtype=torch.bool)
           for key, value in zip(("S2", "S1A", "S1D"), x)}
    with torch.no_grad():
        debug = instance(*x, **dates(), pad_masks=pad, return_debug=True)
    assert debug["window_pad_mask"].all() and debug["temporal_pad_mask"].all()
    assert not debug["pixel_valid"].any()
    assert debug["fused_sequence"].count_nonzero() == 0
    assert temporal(instance).last_inputs.count_nonzero() == 0
    assert torch.isfinite(debug["logits"]).all()
    x[1] = None
    with torch.no_grad():
        debug = instance(*x, **dates(), return_debug=True)
    assert not debug["aligned_valid"]["S1A"].any()
    assert debug["aligned_counts"]["S1A"].count_nonzero() == 0


def test_explicit_validity_ignores_corrupted_cloud_and_padding(mid_module):
    instance = model(mid_module).eval()
    x = inputs()
    pad = {key: torch.zeros(2, value.shape[1], dtype=torch.bool)
           for key, value in zip(("S2", "S1A", "S1D"), x)}
    pad["S1D"][:, -1] = True
    valid = {"S2": torch.ones(2, 3, 1, 16, 16, dtype=torch.bool)}
    valid["S2"][:, 1] = False
    valid["S2"][:, 0, :, :8, :8] = False
    positions = dates()
    positions["batch_positions_s1d"][:, -1] = float("nan")
    altered = [value.clone() for value in x]
    altered[0][~valid["S2"].expand_as(altered[0])] = float("nan")
    altered[2][:, -1] = float("inf")
    with torch.no_grad():
        first = instance(*x, **positions, pad_masks=pad, valid_masks=valid)
        second = instance(*altered, **positions, pad_masks=pad, valid_masks=valid)
    assert torch.isfinite(second).all()
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_explicit_non_padding_zero_frame_is_valid(mid_module):
    instance = model(mid_module).eval()
    x = [torch.zeros_like(value) for value in inputs()]
    pad = {key: torch.zeros(2, value.shape[1], dtype=torch.bool)
           for key, value in zip(("S2", "S1A", "S1D"), x)}
    with torch.no_grad():
        debug = instance(*x, **dates(), pad_masks=pad, return_debug=True)
    assert not debug["window_pad_mask"][:, :3].any()
    assert debug["aligned_counts"]["S2"].sum() > 0


def test_gradients_reach_every_branch_fusion_and_shared_temporal(mid_module):
    instance = model(mid_module).train()
    instance(*inputs(), **dates()).square().mean().backward()
    for branch in instance.spatial_branches.values():
        grad = branch.in_conv.conv.conv[0].weight.grad
        assert grad is not None and grad.count_nonzero() > 0
    shared_grad = temporal(instance).projection.weight.grad
    assert shared_grad is not None and shared_grad.count_nonzero() > 0
    fusion_grad = [item.proj.weight.grad for item in instance.modules()
                   if item.__class__.__name__ == "LateConcatFusion2D"]
    assert fusion_grad and all(grad is not None and grad.count_nonzero() > 0 for grad in fusion_grad)
    assert all(torch.isfinite(p.grad).all() for p in instance.parameters() if p.grad is not None)


@pytest.mark.parametrize("orbit", ["S1A", "S1D"])
def test_single_orbit_interface(mid_module, orbit):
    instance = model(mid_module, sar_orbits=orbit).eval()
    x = inputs()
    x[2 if orbit == "S1A" else 1] = None
    with torch.no_grad():
        debug = instance(*x, **dates(), return_debug=True)
    assert instance.active_modalities == ("S2", orbit)
    assert set(debug["aligned_valid"]) == {"S2", orbit}
    assert debug["logits"].shape == (2, 4, 16, 16)
    temporal(instance)


def test_default_24_windows_and_invalid_configuration(mid_module):
    import inspect

    assert inspect.signature(mid_module.MMUCPMambaMid).parameters["num_bins"].default == 24
    instance = model(mid_module, num_bins=24)
    x = inputs()
    with torch.no_grad():
        debug = instance.eval()(*x, **dates(), return_debug=True)
    assert debug["window_pad_mask"].shape == (2, 24)
    for options in ({"start_day": None, "end_day": None}, {"fusion_mode": "qaf"}):
        with pytest.raises((ValueError, TypeError)):
            model(mid_module, **options)


def test_missing_dates_cannot_silently_use_sequence_index(mid_module):
    instance = model(mid_module)
    with pytest.raises(ValueError):
        instance(*inputs())


def test_cli_exposes_new_model_and_window_parameters():
    tree = ast.parse((ROOT / "train_semantic.py").read_text(encoding="utf-8"))
    arguments = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"]
    options = {arg.value: node for node in arguments for arg in node.args
               if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}
    assert {"--mid-num-bins", "--mid-start-day", "--mid-end-day"} <= set(options)
    choices = next(keyword.value for keyword in options["--model"].keywords if keyword.arg == "choices")
    assert "mm_ucpmamba_mid" in ast.literal_eval(choices)
    default = next(keyword.value for keyword in options["--mid-num-bins"].keywords
                   if keyword.arg == "default")
    assert ast.literal_eval(default) == 24


def test_model_routing_preserves_original_interface(mid_module):
    tree = ast.parse((ROOT / "src/model_utils.py").read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "get_model")
    code = compile(ast.Module(body=[function], type_ignores=[]), "model_utils.py", "exec")
    original_calls = []

    def original_model(**kwargs):
        original_calls.append(kwargs)
        return "original"

    namespace = {"mm_ucpmamba": types.SimpleNamespace(MMUCPMamba=original_model)}
    exec(code, namespace)
    config = types.SimpleNamespace(
        model="mm_ucpmamba_mid", encoder_widths=[8, 8, 8, 16],
        decoder_widths=[8, 8, 8, 16], out_conv=[8, 4],
        str_conv_k=4, str_conv_s=2, str_conv_p=1, encoder_norm="group",
        decoder_norm="group", decoder_groups=8, padding_mode="reflect", pad_value=0,
        mid_num_bins=24, mid_start_day=0, mid_end_day=40,
    )
    routed = namespace["get_model"](config)
    assert isinstance(routed, mid_module.MMUCPMambaMid)
    assert routed.aligner.num_bins == 24
    assert (routed.aligner.start_day, routed.aligner.end_day) == (0, 40)
    config.model = "mm_ucpmamba"
    assert namespace["get_model"](config) == "original"
    assert not {"num_bins", "start_day", "end_day"} & set(original_calls[0])


def test_empty_modality_sequence_is_allowed(mid_module):
    instance = model(mid_module).eval()
    x, positions = inputs(), dates()
    x[1] = x[1][:, :0]
    positions["batch_positions_s1a"] = positions["batch_positions_s1a"][:, :0]
    with torch.no_grad():
        debug = instance(*x, **positions, return_debug=True)
    assert not debug["aligned_valid"]["S1A"].any()
    assert torch.isfinite(debug["logits"]).all()


def test_cpu_autocast_missing_windows_remain_finite(mid_module):
    instance = model(mid_module).eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        debug = instance(*inputs(), **dates(), return_debug=True)
    assert torch.isfinite(debug["logits"]).all()
    assert debug["fused_sequence"][:, -1].count_nonzero() == 0
    assert temporal(instance).last_inputs[:, -1].count_nonzero() == 0
