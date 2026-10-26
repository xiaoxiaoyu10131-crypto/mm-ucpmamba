"""真实 L-TAE 与 1016 主路的数值、缺测、梯度和保存恢复测试。"""

import importlib
import importlib.util
import io
import sys

import pytest
import torch

from convlstm_test_helpers import real_convlstm as _recurrence_fixture
from test_mm_ucpmamba_mid import ROOT, dates, inputs, mid_module as _mid_fixture


mid_module = _mid_fixture
real_convlstm = _recurrence_fixture


@pytest.fixture
def uptae_module(real_convlstm):
    # 共享 fixture 隔离了 CUDA；这里重新加载真实 LTAE，不能沿用它的替身。
    name = "src.backbones.ltae"
    spec = importlib.util.spec_from_file_location(name, ROOT / "src/backbones/ltae.py")
    ltae = importlib.util.module_from_spec(spec)
    sys.modules[name] = ltae
    spec.loader.exec_module(ltae)
    module = importlib.import_module("src.backbones.mm_uptae")
    plain = importlib.import_module("src.backbones.mm_ucpmamba_novmamba")
    return module, ltae, plain


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


def core(module):
    return module.WindowLTAE(input_dim=16, hidden_dim=8, input_size=(2, 2),
                             kernel_size=(3, 3), n_head=4, d_model=32, d_k=4)


def test_only_temporal_encoder_changes_and_all_seven_skips_remain(uptae_module):
    module, _, plain = uptae_module
    original = instance(plain.MMUCPMambaNoVMamba)
    new = instance(module.MMUPTAE, n_head=4, d_model=32, d_k=4)
    outside = lambda model: {k: v.shape for k, v in model.state_dict().items()
                             if not k.startswith("temporal_encoder.")}
    assert outside(new) == outside(original)
    assert isinstance(new.temporal_encoder, module.WindowLTAE)
    assert not any("ConvLSTM" in m.__class__.__name__ for m in new.modules())
    assert not any(".vss." in name or name.endswith(".beta") for name in new.state_dict())
    assert len([m for m in new.modules() if m.__class__.__name__ == "PhenologyAttentionV2"]) == 7
    assert new.out_convlstm.in_channels == 8 and new.out_convlstm.out_channels == 16
    assert new.dynamic_windows.keys() == original.dynamic_windows.keys()
    with pytest.raises((ValueError, TypeError)):
        instance(module.MMUPTAE, temporal_encoder_class=torch.nn.Identity)


def test_full_valid_sequence_matches_original_ltae_with_shared_weights(uptae_module):
    module, ltae, _ = uptae_module
    new = core(module).eval()
    original = ltae.LTAE2d(in_channels=16, n_head=4, d_model=32, d_k=4,
                           mlp=[32, 8], return_att=True).eval()
    original.load_state_dict(new.state_dict())
    torch.manual_seed(4)
    x = torch.randn(2, 4, 16, 2, 2)
    positions = torch.tensor([[0., 7., 21., 35.]]).expand(2, -1)
    with torch.no_grad():
        actual, attention = new(x, batch_positions=positions, return_att=True)
        expected, expected_attention = original(x, batch_positions=positions)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(attention, expected_attention, rtol=1e-5, atol=2e-6)


def test_pixel_mask_matches_individually_unpadded_sequences(uptae_module):
    module, ltae, _ = uptae_module
    new = core(module).eval()
    original = ltae.LTAE2d(in_channels=16, n_head=4, d_model=32, d_k=4,
                           mlp=[32, 8]).eval()
    original.load_state_dict(new.state_dict())
    x = torch.randn(1, 4, 16, 2, 2)
    valid = torch.tensor([[True, False, True, True], [False, True, False, False],
                          [True, True, True, True], [False, False, False, False]])
    valid = valid.T.reshape(1, 4, 1, 2, 2)
    positions = torch.tensor([[1., 7., 21., 35.]])
    dirty = torch.where(valid, x, torch.full_like(x, float("nan")))
    with torch.no_grad():
        actual, attention = new(dirty, batch_positions=positions, valid_mask=valid, return_att=True)
        for row in range(2):
            for col in range(2):
                keep = valid[0, :, 0, row, col]
                if keep.any():
                    compact = x[:, keep, :, row:row + 1, col:col + 1]
                    expected = original(compact, batch_positions=positions[:, keep])
                    torch.testing.assert_close(actual[:, :, row:row + 1, col:col + 1], expected,
                                               rtol=1e-5, atol=2e-6)
                else:
                    assert not actual[:, :, row, col].any()
    invalid = ~valid[:, :, 0].unsqueeze(0).expand_as(attention)
    assert torch.isfinite(actual).all() and attention[invalid].count_nonzero() == 0
    assert not attention[:, :, :, 1, 1].any()


def test_nan_tail_padding_cannot_change_normalization_or_attention(uptae_module):
    module, _, _ = uptae_module
    new = core(module).eval()
    x = torch.randn(2, 3, 16, 2, 2)
    positions = torch.tensor([[1., 12., 28.]]).expand(2, -1)
    padded = torch.cat([x, torch.full_like(x[:, :1], float("nan"))], dim=1)
    padded_dates = torch.cat([positions, torch.full((2, 1), float("nan"))], dim=1)
    pad = torch.tensor([[False, False, False, True]]).expand(2, -1)
    with torch.no_grad():
        expected = new(x, batch_positions=positions)
        actual, attention = new(padded, batch_positions=padded_dates, pad_mask=pad, return_att=True)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
    assert attention[:, :, -1].count_nonzero() == 0


@pytest.mark.parametrize("training", [False, True])
def test_all_invalid_and_one_active_pixel_are_finite(uptae_module, training):
    module, _, _ = uptae_module
    new = core(module).train(training)
    x = torch.randn(1, 4, 16, 2, 2, requires_grad=True)
    positions = torch.tensor([[1., 12., 28., 35.]])
    valid = torch.zeros(1, 4, 1, 2, 2, dtype=torch.bool)
    empty, attention = new(x, batch_positions=positions, valid_mask=valid, return_att=True)
    assert torch.isfinite(empty).all() and not empty.any() and not attention.any()
    valid[0, 0, 0, 0, 0] = True
    output = new(x, batch_positions=positions, valid_mask=valid)
    assert torch.isfinite(output).all() and not output[:, :, 1, 1].any()
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert not x.grad[~valid.expand_as(x)].any()


def test_full_model_passes_window_centers_and_pixel_validity(uptae_module):
    module, _, _ = uptae_module
    new = instance(module.MMUPTAE, n_head=4, d_model=32, d_k=4).eval()
    received = {}

    def capture(_module, args, kwargs):
        received.update(kwargs)

    handle = new.temporal_encoder.register_forward_pre_hook(capture, with_kwargs=True)
    with torch.no_grad():
        debug = new(*inputs(), **dates(), return_debug=True)
    handle.remove()
    expected = new.aligner.centers[None].expand(2, -1)
    torch.testing.assert_close(received["batch_positions"], expected)
    assert received["valid_mask"].shape == (2, 4, 1, 2, 2)
    assert not received["valid_mask"][:, -1].any()
    assert debug["logits"].shape == (2, 4, 16, 16)
    assert torch.isfinite(debug["logits"]).all()


def test_empty_pixels_do_not_change_training_batchnorm_statistics(uptae_module):
    module, _, _ = uptae_module
    normal, extended = core(module).train(), core(module).train()
    extended.load_state_dict(normal.state_dict())
    x = torch.randn(1, 4, 16, 1, 2)
    padded = torch.cat([x, torch.full_like(x[..., :1], float("nan"))], dim=-1)
    valid = torch.ones(1, 4, 1, 1, 3, dtype=torch.bool)
    valid[..., -1] = False
    positions = torch.tensor([[1., 12., 28., 35.]])
    torch.manual_seed(22)
    expected = normal(x, batch_positions=positions)
    torch.manual_seed(22)
    actual = extended(padded, batch_positions=positions, valid_mask=valid)
    torch.testing.assert_close(actual[..., :2], expected, rtol=1e-5, atol=2e-6)
    assert not actual[..., -1].any()
    for name, value in normal.state_dict().items():
        if "running_" in name or name.endswith("num_batches_tracked"):
            torch.testing.assert_close(extended.state_dict()[name], value, rtol=0, atol=0)


def test_low_memory_training_matches_standard_outputs_and_ltae_gradients(uptae_module):
    module, _, _ = uptae_module
    normal = instance(module.MMUPTAE, n_head=4, d_model=32, d_k=4).train()
    low = instance(module.MMUPTAE, n_head=4, d_model=32, d_k=4, low_memory=True).train()
    low.load_state_dict(normal.state_dict())
    data, positions = inputs(), dates()
    torch.manual_seed(41)
    first = normal(*data, **positions)
    torch.manual_seed(41)
    second = low(*data, **positions)
    torch.testing.assert_close(first, second, rtol=1e-5, atol=2e-6)
    first.square().mean().backward()
    second.square().mean().backward()
    torch.testing.assert_close(normal.temporal_encoder.attention_heads.Q.grad,
                               low.temporal_encoder.attention_heads.Q.grad,
                               rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize("low_memory", [False, True])
def test_two_updates_reach_ltae_and_all_retained_components(uptae_module, low_memory):
    module, _, _ = uptae_module
    new = instance(module.MMUPTAE, n_head=4, d_model=32, d_k=4, low_memory=low_memory).train()
    optimizer = torch.optim.Adam(new.parameters(), lr=.01)
    for _ in range(2):
        optimizer.zero_grad()
        new(*inputs(), **dates()).square().mean().backward()
        optimizer.step()
    for prefix in ("temporal_encoder", "spatial_branches", "dynamic_windows", "async_interaction",
                   "mid_fusion", "temporal_aggregator", "up_blocks", "out_convlstm"):
        grads = [p.grad for n, p in new.named_parameters() if n.startswith(prefix) and p.grad is not None]
        assert grads and sum(g.abs().sum() for g in grads) > 0, prefix
    assert all(torch.isfinite(p.grad).all() for p in new.parameters() if p.grad is not None)


def test_full_model_handles_missing_modality_and_all_invalid_windows(uptae_module):
    module, _, _ = uptae_module
    new = instance(module.MMUPTAE, n_head=4, d_model=32, d_k=4).eval()
    data = inputs()
    valid = {name: torch.zeros(value.shape[:2], dtype=torch.bool)
             for name, value in zip(("S2", "S1A", "S1D"), data)}
    with torch.no_grad():
        empty = new(*data, **dates(), valid_masks=valid, return_debug=True)
        data[1] = None
        missing = new(*data, **dates(), return_debug=True)
    assert torch.isfinite(empty["logits"]).all() and empty["temporal_pad_mask"].all()
    assert torch.isfinite(missing["logits"]).all() and not missing["aligned_valid"]["S1A"].any()


def test_checkpoint_roundtrip_and_calendar_isolation(uptae_module):
    module, _, plain = uptae_module
    new = instance(module.MMUPTAE, n_head=4, d_model=32, d_k=4).eval()
    buffer = io.BytesIO()
    torch.save(new.state_dict(), buffer)
    buffer.seek(0)
    state = torch.load(buffer, weights_only=True)
    restored = instance(module.MMUPTAE, n_head=4, d_model=32, d_k=4).eval()
    restored.load_state_dict(state)
    with torch.no_grad():
        torch.testing.assert_close(new(*inputs(), **dates()), restored(*inputs(), **dates()),
                                   rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="date windows"):
        instance(module.MMUPTAE, n_head=4, d_model=32, d_k=4, end_day=41).load_state_dict(state)
    with pytest.raises(RuntimeError):
        instance(plain.MMUCPMambaNoVMamba).load_state_dict(state)
