"""复用原有CPU时序替身，仅验证新模块与真实空间/解码组件的接线。"""

import importlib
import io

import pytest
import torch

from test_mm_ucpmamba_mid import dates, inputs, mid_module as _mid_fixture, model

mid_module = _mid_fixture


def instance(**options):
    module = importlib.import_module("src.backbones.mm_ucpmamba_async")
    proxy = type("Factory", (), {"MMUCPMambaMid": module.MMUCPMambaAsync})
    return model(proxy, **options)


def test_zero_residual_matches_108_wiring_exactly(mid_module):
    baseline, new = model(mid_module).eval(), instance().eval()
    new.load_state_dict(baseline.state_dict(), strict=False)
    with torch.no_grad():
        a = baseline(*inputs(), **dates(), return_debug=True)
        b = new(*inputs(), **dates(), return_debug=True)
    torch.testing.assert_close(a["logits"], b["logits"], rtol=0, atol=0)
    torch.testing.assert_close(a["fused_sequence"], b["fused_sequence"], rtol=0, atol=0)
    assert torch.equal(a["pixel_valid"], b["pixel_valid"])
    assert len([m for m in new.modules() if m.__class__.__name__ == "PhenologyAttentionV2"]) == 7


def test_missing_modality_no_nan_and_no_pseudo_observations(mid_module):
    new = instance().eval()
    data = inputs()
    data[1] = None
    with torch.no_grad():
        debug = new(*data, **dates(), return_debug=True)
    assert torch.isfinite(debug["logits"]).all()
    assert not debug["aligned_valid"]["S1A"].any()
    assert not debug["alignment_info"]["S1A"]["info"].any()


def test_two_updates_reach_new_modules_shared_temporal_and_decoder(mid_module):
    new = instance().train()
    optimizer = torch.optim.Adam(new.parameters(), lr=.01)
    for _ in range(2):
        optimizer.zero_grad()
        loss = new(*inputs(), **dates()).square().mean()
        loss.backward()
        optimizer.step()
    for prefix in ("spatial_branches", "dynamic_windows", "async_interaction", "temporal_encoder", "up_blocks"):
        grads = [p.grad for n, p in new.named_parameters() if n.startswith(prefix) and p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0


def test_checkpoint_roundtrip_and_reject_changed_calendar_or_async_config(mid_module):
    new = instance().eval()
    buffer = io.BytesIO()
    torch.save(new.state_dict(), buffer)
    buffer.seek(0)
    restored = instance().eval()
    state = torch.load(buffer, weights_only=True)
    restored.load_state_dict(state)
    with torch.no_grad():
        torch.testing.assert_close(new(*inputs(), **dates()), restored(*inputs(), **dates()), rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="date windows"):
        instance(end_day=41).load_state_dict(state)
    with pytest.raises(RuntimeError, match="async configuration"):
        instance(trend_min_span=3).load_state_dict(state)


def test_low_memory_and_post_initialization_reset(mid_module):
    from src.learning.weight_init import weight_init

    new = instance(low_memory=True)
    new.apply(weight_init)
    new.reset_async_initialization()
    assert all(m.gamma == 0 for m in new.dynamic_windows.values())
    assert all(g == 0 for g in new.async_interaction.gamma.values())
    loss = new(*inputs(), **dates()).square().mean()
    loss.backward()
    assert torch.isfinite(loss)


def test_all_modalities_invalid_never_create_valid_windows(mid_module):
    new = instance().eval()
    data = [torch.full_like(x, float("nan")) for x in inputs()]
    valid = {name: torch.zeros(x.shape[:2], dtype=torch.bool)
             for name, x in zip(("S2", "S1A", "S1D"), data)}
    times = {key: torch.full_like(value, float("nan")) for key, value in dates().items()}
    with torch.no_grad():
        debug = new(*data, **times, valid_masks=valid, return_debug=True)
    assert torch.isfinite(debug["logits"]).all()
    assert not debug["pixel_valid"].any() and not debug["fused_sequence"].any()
    assert debug["temporal_pad_mask"].all()


def test_dtype_conversion_preserves_date_and_configuration_precision(mid_module):
    new = instance(trend_min_span=2.123456789).half()
    assert new.async_signature.dtype == torch.float64
    assert new.async_signature[2] == 2.123456789
    assert new.aligner.edges.dtype == torch.float64
