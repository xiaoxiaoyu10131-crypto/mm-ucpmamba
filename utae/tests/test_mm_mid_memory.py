"""低显存路线的数值、梯度和 CPU 图保留张量验证；不验证 CUDA 峰值。"""

import ast
import copy
import json
import sys
import types

import pytest
import torch

from test_mm_ucpmamba_mid import ROOT, model
from test_mm_ucpmamba_mid import mid_module as _mid_module


@pytest.fixture
def mid_module(monkeypatch):
    yield from _mid_module.__wrapped__(monkeypatch)


def long_batch(requires_grad=False):
    torch.manual_seed(81)
    inputs = [torch.randn(2, t, c, 16, 16, requires_grad=requires_grad)
              for t, c in [(17, 10), (12, 3), (9, 3)]]
    names = ("S2", "S1A", "S1D")
    dates = {"batch_positions_" + key.lower(): torch.linspace(1, 39, x.shape[1]).repeat(2, 1)
             for key, x in zip(names, inputs)}
    pads = {key: torch.zeros(2, x.shape[1], dtype=torch.bool) for key, x in zip(names, inputs)}
    for pad in pads.values():
        pad[0, -1] = True
    valid = {"S2": torch.ones(2, 17, 1, 16, 16, dtype=torch.bool)}
    valid["S2"][0, 4] = False
    valid["S2"][1, 7:10, :, :8, :8] = False
    return inputs, dict(**dates, pad_masks=pads, valid_masks=valid)


def assert_gradients_close(first, second):
    first_params, second_params = dict(first.named_parameters()), dict(second.named_parameters())
    assert first_params.keys() == second_params.keys()
    for name, parameter in first_params.items():
        other = second_params[name]
        assert (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            assert torch.isfinite(other.grad).all(), name
            torch.testing.assert_close(parameter.grad, other.grad, rtol=3e-3, atol=5e-6, msg=name)


def test_low_memory_model_preserves_outputs_and_all_gradients(mid_module):
    first, second = model(mid_module).train(), model(mid_module, low_memory=True).train()
    second.load_state_dict(first.state_dict(), strict=True)
    assert first.state_dict().keys() == second.state_dict().keys()
    x, options = long_batch(requires_grad=True)
    y = [item.detach().clone().requires_grad_() for item in x]
    first_output, second_output = first(*x, **options), second(*y, **options)
    torch.testing.assert_close(first_output, second_output, rtol=5e-4, atol=2e-5)
    first_output.square().mean().backward()
    second_output.square().mean().backward()
    assert_gradients_close(first, second)
    for a, b in zip(x, y):
        torch.testing.assert_close(a.grad, b.grad, rtol=3e-3, atol=5e-6)


def test_checkpoint_trains_parameters_without_input_gradients(mid_module):
    instance = model(mid_module, low_memory=True).train()
    x, options = long_batch()
    assert all(not item.requires_grad for item in x)
    instance(*x, **options).square().mean().backward()
    for branch in instance.spatial_branches.values():
        assert branch.low_memory
        for layer in [branch.in_conv, *branch.down_blocks, *branch.temporal_aggregator]:
            gradients = [p.grad for p in layer.parameters() if p.requires_grad]
            assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
            assert any(g.count_nonzero() > 0 for g in gradients)


@pytest.mark.parametrize("chunk_size", [4, 8])
def test_phenology_chunk_halo_matches_values_and_gradients(mid_module, chunk_size):
    components = sys.modules["src.backbones.mm_mid_components"]
    torch.manual_seed(84)
    first = components.PhenologyAttentionV2(8, stage_level=0)
    second = copy.deepcopy(first)
    x = torch.randn(2, 17, 8, 8, 8, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    valid = torch.rand(2, 17, 1, 8, 8) > 0.2
    valid[0, 5] = False
    valid[1, :, :, 0, 0] = False
    a = components.masked_phenology(first, x, valid)
    b = components.masked_phenology(second, y, valid, time_chunk_size=chunk_size)
    torch.testing.assert_close(a, b, rtol=5e-5, atol=2e-6)
    a.square().mean().backward()
    b.square().mean().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=1e-4, atol=1e-6)
    assert_gradients_close(first, second)
    assert second.conv3d.weight.grad.count_nonzero() > 0
    assert second.alpha.grad.abs() > 0
    assert b[1, :, 0, 0].count_nonzero() == 0


def test_frame_chunks_preserve_group_norm_and_padding(mid_module):
    components = sys.modules["src.backbones.mm_mid_components"]
    first = components.ConvBlock([3, 8, 8], norm="group", pad_value=None)
    second = copy.deepcopy(first)
    x = torch.randn(2, 17, 3, 8, 8, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    pad = torch.zeros(2, 17, dtype=torch.bool)
    pad[0, 7:10] = True
    a = components.framewise_forward(first, x, pad, 8, (8, 8))
    b = components.framewise_forward(second, y, pad, 8, (8, 8), chunk_size=8)
    torch.testing.assert_close(a, b, rtol=5e-5, atol=1e-6)
    assert b[0, 7:10].count_nonzero() == 0
    a.square().mean().backward()
    b.square().mean().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=1e-4, atol=1e-6)
    assert_gradients_close(first, second)


@pytest.mark.parametrize("empty_length", [0, 17])
def test_empty_branch_and_all_missing_model_stay_zero(mid_module, empty_length):
    instance = model(mid_module, low_memory=True).train()
    x = torch.randn(2, empty_length, 10, 16, 16)
    valid = torch.zeros(2, empty_length, 1, 16, 16, dtype=torch.bool)
    feature, mask, skips, skip_valid, pad = instance.s2_branch(x, valid)
    assert feature.shape == (2, empty_length, 8, 4, 4)
    assert feature.count_nonzero() == 0 and not mask.any()
    assert all(skip.count_nonzero() == 0 for skip in skips)
    assert all(not value.any() for value in skip_valid)
    assert pad.all()
    inputs, options = long_batch()
    options["pad_masks"] = {name: torch.ones_like(value) for name, value in options["pad_masks"].items()}
    debug = instance(*inputs, **options, return_debug=True)
    assert debug["window_pad_mask"].all() and debug["temporal_pad_mask"].all()
    assert debug["fused_sequence"].count_nonzero() == 0
    assert torch.isfinite(debug["logits"]).all()


def test_eval_and_no_grad_do_not_checkpoint(mid_module, monkeypatch):
    components = sys.modules["src.backbones.mm_mid_components"]
    calls = []
    checkpoint = components.checkpoint

    def tracked(*args, **kwargs):
        calls.append(kwargs)
        return checkpoint(*args, **kwargs)

    monkeypatch.setattr(components, "checkpoint", tracked)
    instance = model(mid_module, low_memory=True)
    x, options = long_batch()
    instance.eval()(*x, **options)
    with torch.no_grad():
        instance.train()(*x, **options)
    assert not calls
    instance.train()(*x, **options)
    assert len(calls) >= 3
    assert all(call.get("use_reentrant") is False for call in calls)


def test_low_memory_rejects_running_statistics_norm(mid_module):
    with pytest.raises(ValueError, match="(?i)batch|running|norm"):
        model(mid_module, low_memory=True, encoder_norm="batch")
    model(mid_module, low_memory=False, encoder_norm="batch")


def test_checkpoint_reduces_cpu_saved_activation_bytes(mid_module):
    first = model(mid_module).s2_branch.train()
    second = model(mid_module, low_memory=True).s2_branch.train()
    second.load_state_dict(first.state_dict())
    x, _ = long_batch()
    valid = torch.ones(2, 17, 1, 16, 16, dtype=torch.bool)

    def saved_bytes(branch):
        parameter_storages = {p.untyped_storage().data_ptr() for p in branch.parameters()}
        total = 0

        def pack(tensor):
            nonlocal total
            if tensor.untyped_storage().data_ptr() not in parameter_storages:
                total += tensor.numel() * tensor.element_size()
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            output = branch(x[0], valid)
        assert output[0].requires_grad
        return total

    baseline, optimized = saved_bytes(first), saved_bytes(second)
    assert baseline > 0 and optimized < baseline * 0.65


def test_low_memory_cpu_autocast_forward_backward_is_finite(mid_module):
    instance = model(mid_module, low_memory=True).train()
    x, options = long_batch()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = instance(*x, **options)
        loss = output.float().square().mean()
    loss.backward()
    assert torch.isfinite(output).all()
    assert all(torch.isfinite(p.grad).all() for p in instance.parameters() if p.grad is not None)
    assert instance.s2_branch.in_conv.conv.conv[0].weight.grad.count_nonzero() > 0


def test_low_memory_cli_defaults_and_model_routing(mid_module, monkeypatch):
    tree = ast.parse((ROOT / "train_semantic.py").read_text(encoding="utf-8"))
    argument = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                    and any(isinstance(arg, ast.Constant) and arg.value == "--mid-low-memory"
                            for arg in node.args))
    action = next(keyword.value for keyword in argument.keywords if keyword.arg == "action")
    assert ast.literal_eval(action) == "store_true"
    if any(keyword.arg == "default" for keyword in argument.keywords):
        assert ast.literal_eval(next(keyword.value for keyword in argument.keywords
                                     if keyword.arg == "default")) is False
    tree = ast.parse((ROOT / "src/model_utils.py").read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "get_model")
    original_calls = []
    namespace = {"mm_ucpmamba": types.SimpleNamespace(
        MMUCPMamba=lambda **kwargs: original_calls.append(kwargs))}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "model_utils.py", "exec"), namespace)
    config = types.SimpleNamespace(
        model="mm_ucpmamba_mid", encoder_widths=[8, 8, 8, 16], decoder_widths=[8, 8, 8, 16],
        out_conv=[8, 4], str_conv_k=4, str_conv_s=2, str_conv_p=1, encoder_norm="group",
        decoder_norm="group", decoder_groups=8, padding_mode="reflect", pad_value=0,
        mid_start_day=0, mid_end_day=40, mid_low_memory=True,
    )
    routed = namespace["get_model"](config)
    assert all(branch.low_memory for branch in routed.spatial_branches.values())
    config.model = "mm_ucpmamba"
    monkeypatch.setattr(sys.modules["src.backbones.mm_ucpmamba"], "MMUCPMamba",
                        lambda **kwargs: original_calls.append(kwargs))
    namespace["get_model"](config)
    assert "low_memory" not in original_calls[0]


def test_batch_four_low_memory_optimizer_step_is_finite(mid_module):
    instance = model(mid_module, low_memory=True).train()
    x, options = long_batch()
    x = [torch.cat([item, item], dim=0) for item in x]
    options = {key: ({name: value.repeat(2, *([1] * (value.ndim - 1)))
                     for name, value in item.items()} if isinstance(item, dict)
                     else item.repeat(2, 1)) for key, item in options.items()}
    optimizer = torch.optim.SGD(instance.parameters(), lr=0.01)
    previous = {name: p.detach().clone() for name, p in instance.named_parameters()}
    output = instance(*x, **options)
    output.square().mean().backward()
    optimizer.step()
    assert output.shape == (4, 4, 16, 16) and torch.isfinite(output).all()
    assert all(not item.requires_grad for item in x)
    assert all(torch.isfinite(p).all() for p in instance.parameters())
    assert any(not torch.equal(p.detach(), previous[name]) for name, p in instance.named_parameters())


def test_two_steps_preserve_final_batch_norm_running_statistics(mid_module):
    first, second = model(mid_module).train(), model(mid_module, low_memory=True).train()
    second.load_state_dict(first.state_dict())
    optimizers = [torch.optim.SGD(item.parameters(), lr=0.003) for item in [first, second]]
    for _ in range(2):
        x, options = long_batch()
        for instance, optimizer in zip([first, second], optimizers):
            optimizer.zero_grad(set_to_none=True)
            instance(*x, **options).square().mean().backward()
            optimizer.step()
    first_bn = [item for item in first.out_conv.modules() if isinstance(item, torch.nn.BatchNorm2d)]
    second_bn = [item for item in second.out_conv.modules() if isinstance(item, torch.nn.BatchNorm2d)]
    assert first_bn and len(first_bn) == len(second_bn)
    for a, b in zip(first_bn, second_bn):
        assert a.num_batches_tracked.item() == b.num_batches_tracked.item() == 2
        torch.testing.assert_close(a.running_mean, b.running_mean, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(a.running_var, b.running_var, rtol=1e-4, atol=1e-6)


def test_oom_context_and_failed_cuda_diagnostics_preserve_original_error(mid_module, monkeypatch, capsys):
    report = sys.modules["src.backbones.mm_mid_memory"].report_mid_oom
    x, options = long_batch()
    inputs = dict(zip(["S2", "S1A", "S1D"], x), pad_masks=options["pad_masks"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    report(inputs, torch.device("cpu"))
    details = json.loads(capsys.readouterr().out.split("context: ", 1)[1])
    assert details["sequence_shapes"] == {name: list(value.shape) for name, value in zip(["S2", "S1A", "S1D"], x)}
    assert details["non_padding_frames"] == {"S2": [16, 17], "S1A": [11, 12], "S1D": [8, 9]}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def failed_cuda(*args):
        raise RuntimeError("CUDA diagnostics failed")

    monkeypatch.setattr(torch.cuda, "mem_get_info", failed_cuda)
    with pytest.raises(torch.cuda.OutOfMemoryError, match="original allocation failure"):
        try:
            raise torch.cuda.OutOfMemoryError("original allocation failure")
        except torch.cuda.OutOfMemoryError:
            report(inputs, torch.device("cuda:0"))
            raise
    assert "CUDA diagnostics failed" in capsys.readouterr().out
