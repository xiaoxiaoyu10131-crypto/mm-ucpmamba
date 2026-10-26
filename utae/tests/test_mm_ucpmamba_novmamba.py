"""无 VMamba 的真实普通门控与 1016 主路回归；仅隔离 CUDA VSS 导入。"""

import importlib
import io

import pytest
import torch

from test_mm_ucpmamba_mid import dates, inputs, mid_module as _mid_fixture
from convlstm_test_helpers import real_convlstm as _real_recurrence_fixture


mid_module = _mid_fixture
real_convlstm = _real_recurrence_fixture


@pytest.fixture
def plain_module(real_convlstm):
    module = importlib.import_module("src.backbones.mm_ucpmamba_novmamba")
    async_module = importlib.import_module("src.backbones.mm_ucpmamba_async")
    plain = importlib.import_module("src.backbones.uconvlstm")
    return module, async_module, real_convlstm, plain


def instance(module, **kwargs):
    options = dict(
        optical_input_dim=10, sar_input_dim=3,
        encoder_widths=[8, 8, 8, 16], decoder_widths=[8, 8, 8, 16],
        out_conv=[8, 4], input_size=16, hidden_dim=8,
        decoder_norm="group", decoder_groups=8,
        num_bins=4, start_day=0, end_day=40,
    )
    options.update(kwargs)
    return module(**options)


def paired_models(plain_module, **kwargs):
    module, async_module, convlstm, _ = plain_module
    baseline = instance(async_module.MMUCPMambaAsync,
                        temporal_encoder_class=convlstm.ConvLSTM, **kwargs)
    new = instance(module.MMUCPMambaNoVMamba, **kwargs)
    state = baseline.state_dict()
    new.load_state_dict({name: state[name] for name in new.state_dict()})
    return baseline, new


def test_only_vss_parameters_disappear_and_seven_phenology_modules_remain(plain_module):
    _, async_module, _, plain = plain_module
    baseline, new = paired_models(plain_module)
    assert isinstance(new, async_module.MMUCPMambaAsync)
    assert isinstance(new.temporal_encoder, plain.PlainConvLSTM)
    assert isinstance(new.temporal_encoder.cell_list[0], plain.PlainConvLSTMCell)
    assert not any("vss" in name.lower() or name.endswith("beta")
                   for name, _ in new.named_parameters())
    old_shapes = {name: value.shape for name, value in baseline.state_dict().items()}
    new_shapes = {name: value.shape for name, value in new.state_dict().items()}
    removed = old_shapes.keys() - new_shapes.keys()
    assert removed and all(name.startswith("temporal_encoder.cell_list.0.")
                           and (".vss." in name or name.endswith(".beta")) for name in removed)
    assert new_shapes == {name: shape for name, shape in old_shapes.items() if name not in removed}
    assert len([m for m in new.modules() if m.__class__.__name__ == "PhenologyAttentionV2"]) == 7
    assert new.out_convlstm.in_channels == 8 and new.out_convlstm.out_channels == 16
    assert set(new.dynamic_windows) == {"S2", "S1A", "S1D"}
    assert new.modality_dropout == 0 and new.fusion_mode == "concat"


@pytest.mark.parametrize("low_memory", [False, True])
def test_shared_weights_match_beta_zero_baseline_logits_and_masks(plain_module, low_memory):
    baseline, new = paired_models(plain_module, low_memory=low_memory)
    baseline.eval()
    new.eval()
    assert baseline.temporal_encoder.cell_list[0].beta.item() == 0
    with torch.no_grad():
        expected = baseline(*inputs(), **dates(), return_debug=True)
        actual = new(*inputs(), **dates(), return_debug=True)
    for key in ("logits", "fused_sequence", "temporal_pad_mask", "pixel_valid"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert actual["logits"].shape == (2, 4, 16, 16)
    assert actual["temporal_input_shape"] == (2, 4, 16, 2, 2)
    assert actual["temporal_pad_mask"].tolist() == [[False, False, False, True]] * 2


@pytest.mark.parametrize("low_memory", [False, True])
def test_plain_gate_and_retained_async_components_receive_finite_gradients(plain_module, low_memory):
    module, _, _, _ = plain_module
    new = instance(module.MMUCPMambaNoVMamba, low_memory=low_memory).train()
    optimizer = torch.optim.Adam(new.parameters(), lr=.01)
    for _ in range(2):
        optimizer.zero_grad()
        new(*inputs(), **dates()).square().mean().backward()
        optimizer.step()
    gate = new.temporal_encoder.cell_list[0].conv.weight
    assert gate.grad is not None and gate.grad.count_nonzero() > 0
    for prefix in ("spatial_branches", "mid_fusion", "dynamic_windows", "async_interaction",
                   "temporal_aggregator", "up_blocks", "out_convlstm"):
        grads = [p.grad for name, p in new.named_parameters()
                 if name.startswith(prefix) and p.grad is not None]
        assert grads and sum(g.abs().sum() for g in grads) > 0, prefix
    assert all(torch.isfinite(p.grad).all() for p in new.parameters() if p.grad is not None)


def test_plain_recurrence_skips_internal_padding_and_all_padding(plain_module):
    _, _, _, plain = plain_module
    temporal = plain.PlainConvLSTM((2, 2), 4, 3, (3, 3)).eval()
    sequence = torch.randn(2, 5, 4, 2, 2)
    pad = torch.tensor([[False, True, False, True, False], [True, False, True, False, True]])
    sequence[pad] = 123.
    with torch.no_grad():
        _, padded = temporal(sequence, pad_mask=pad)
        for sample in range(2):
            _, expected = temporal(sequence[sample:sample + 1, ~pad[sample]])
            for state in (0, 1):
                torch.testing.assert_close(padded[0][state][sample:sample + 1], expected[0][state])
        _, empty = temporal(sequence, pad_mask=torch.ones_like(pad))
    assert all(torch.isfinite(state).all() and state.count_nonzero() == 0 for state in empty[0])


def test_appended_nan_padding_does_not_change_logits(plain_module):
    module, _, _, _ = plain_module
    new = instance(module.MMUCPMambaNoVMamba).eval()
    data, positions = inputs(), dates()
    padded, masks, extended_dates = [], {}, {}
    for name, key, value in zip(("S2", "S1A", "S1D"), positions, data):
        padded.append(torch.cat([value, torch.full_like(value[:, :1], float("nan"))], dim=1))
        masks[name] = torch.zeros(2, value.shape[1] + 1, dtype=torch.bool)
        masks[name][:, -1] = True
        times = positions[key]
        extended_dates[key] = torch.cat([times, torch.full_like(times[:, :1], float("nan"))], dim=1)
    with torch.no_grad():
        expected = new(*data, **positions)
        actual = new(*padded, **extended_dates, pad_masks=masks)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_missing_modality_and_all_invalid_windows_stay_finite(plain_module):
    module, _, _, _ = plain_module
    new = instance(module.MMUCPMambaNoVMamba).eval()
    data = inputs()
    valid = {name: torch.zeros(value.shape[:2], dtype=torch.bool)
             for name, value in zip(("S2", "S1A", "S1D"), data)}
    with torch.no_grad():
        debug = new(*data, **dates(), valid_masks=valid, return_debug=True)
    assert torch.isfinite(debug["logits"]).all()
    assert debug["temporal_pad_mask"].all() and not debug["fused_sequence"].any()
    data[1] = None
    with torch.no_grad():
        debug = new(*data, **dates(), return_debug=True)
    assert torch.isfinite(debug["logits"]).all() and not debug["aligned_valid"]["S1A"].any()


def test_low_memory_training_matches_standard_outputs_and_gate_gradients(plain_module):
    module, _, _, _ = plain_module
    normal = instance(module.MMUCPMambaNoVMamba, low_memory=False).train()
    low = instance(module.MMUCPMambaNoVMamba, low_memory=True).train()
    low.load_state_dict(normal.state_dict())
    first, second = normal(*inputs(), **dates()), low(*inputs(), **dates())
    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)
    first.square().mean().backward()
    second.square().mean().backward()
    torch.testing.assert_close(normal.temporal_encoder.cell_list[0].conv.weight.grad,
                               low.temporal_encoder.cell_list[0].conv.weight.grad)


def test_checkpoint_roundtrip_calendar_defaults_and_strict_structure_isolation(plain_module):
    module, _, _, plain = plain_module
    baseline, new = paired_models(plain_module)
    new.eval()
    buffer = io.BytesIO()
    torch.save(new.state_dict(), buffer)
    buffer.seek(0)
    restored = instance(module.MMUCPMambaNoVMamba).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(new(*inputs(), **dates()), restored(*inputs(), **dates()),
                                   rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        new.load_state_dict(baseline.state_dict())
    with pytest.raises(RuntimeError, match="Missing key"):
        baseline.load_state_dict(new.state_dict())
    default = module.MMUCPMambaNoVMamba(
        encoder_widths=(8, 8, 8, 16), decoder_widths=(8, 8, 8, 16),
        out_conv=(8, 4), input_size=16, hidden_dim=8,
    )
    assert isinstance(default.temporal_encoder, plain.PlainConvLSTM)
    assert default.aligner.num_bins == 24
    assert (default.aligner.start_day, default.aligner.end_day) == (-200, 600)
