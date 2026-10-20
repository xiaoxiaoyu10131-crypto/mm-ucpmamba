"""邻窗候选、真实日期约束和无伪观测行为。"""

import torch

from src.backbones.mm_async_interaction import LocalAsyncInteraction


def setup():
    names = ("S2", "S1A", "S1D")
    features = {n: torch.randn(1, 4, 8, 2, 2, requires_grad=True) for n in names}
    valid = {n: torch.ones(1, 4, 1, 2, 2, dtype=torch.bool) for n in names}
    stats = {n: {"mean_dates": torch.arange(4).double().view(1, 4, 1, 1, 1).expand(1, 4, 1, 2, 2) * 10,
                 "info": torch.ones(1, 4, 5, 2, 2)} for n in names}
    return names, features, valid, stats


def test_zero_gamma_is_identity_and_all_empty_candidates_safe():
    names, features, valid, stats = setup()
    module = LocalAsyncInteraction(names, channels=8)
    out, diagnostic = module(features, valid, stats, 10.)
    for n in names:
        torch.testing.assert_close(out[n], features[n], rtol=0, atol=0)
    assert diagnostic["candidate_count"] == 240
    for n in names[1:]:
        valid[n].zero_()
        features[n] = torch.full_like(features[n], float("nan"))
    with torch.no_grad():
        for gamma in module.gamma.values():
            gamma.fill_(1)
    out, diagnostic = module(features, valid, stats, 10.)
    torch.testing.assert_close(out["S2"], features["S2"], rtol=0, atol=0)
    assert not out["S1A"].any() and not out["S1D"].any()
    assert diagnostic["active_queries"] == 0
    assert all(torch.isfinite(x).all() for x in out.values())


def test_real_time_penalty_prefers_closer_dates_and_only_neighbor_windows():
    names, features, valid, stats = setup()
    module = LocalAsyncInteraction(names, channels=8, dim=1, radius=1)
    with torch.no_grad():
        for p in module.parameters():
            p.zero_()
        module.gamma["S2"].fill_(1)
        module.value["S1A"].bias.fill_(1)
        module.output["S2"].weight.fill_(1)
    for n in names:
        valid[n].zero_()
    valid["S2"][:, 1] = True
    valid["S1A"][:, 2] = True
    valid["S1D"][:, 1] = True
    # 同箱 S1D 的实际日期更远，邻箱 S1A 只有一天差。
    stats["S1D"]["mean_dates"] = stats["S1D"]["mean_dates"].clone()
    stats["S1D"]["mean_dates"][:, 1] = 1
    stats["S1A"]["mean_dates"] = stats["S1A"]["mean_dates"].clone()
    stats["S1A"]["mean_dates"][:, 2] = 11
    out, _ = module(features, valid, stats, 10.)
    residual = out["S2"][:, 1] - features["S2"][:, 1]
    assert (residual > .5).all()
    valid["S1A"][:, 2] = False
    valid["S1A"][:, 3] = True  # 超出半径，不可访问
    out, _ = module(features, valid, stats, 10.)
    torch.testing.assert_close(out["S2"][:, 1], features["S2"][:, 1])
    assert not out["S2"][:, 0].any()


def test_nonzero_residual_gradients_reach_qkv_aux_time_and_features():
    names, features, valid, stats = setup()
    module = LocalAsyncInteraction(names, channels=8)
    with torch.no_grad():
        for gamma in module.gamma.values():
            gamma.fill_(.1)
    out, _ = module(features, valid, stats, 10.)
    sum(x.square().mean() for x in out.values()).backward()
    for name, p in module.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name
    assert all(x.grad.abs().sum() > 0 for x in features.values())


def test_parameter_budget_exact():
    from src.backbones.mm_dynamic_window import DynamicWindowRepresentation

    assert 3 * sum(p.numel() for p in DynamicWindowRepresentation().parameters()) == 6627
    assert sum(p.numel() for p in LocalAsyncInteraction().parameters()) == 12738
