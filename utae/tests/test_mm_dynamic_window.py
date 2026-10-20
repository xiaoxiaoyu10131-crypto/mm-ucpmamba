"""真实日期统计的数值、掩码、边界和梯度单元测试。"""

import pytest
import torch

from src.backbones.mm_dynamic_window import DynamicWindowRepresentation
from src.backbones.mm_temporal_alignment import DateWindowAligner


def run(values, dates, valid=None, min_span=2):
    x = torch.tensor(values, dtype=torch.float32).reshape(1, -1, 1, 1, 1).requires_grad_()
    t = torch.tensor([dates], dtype=torch.float64)
    pad = torch.zeros(t.shape, dtype=torch.bool)
    module = DynamicWindowRepresentation(1, min_span=min_span)
    captured = []
    module.projection[0].register_forward_pre_hook(lambda _, inputs: captured.append(inputs[0]))
    output = module(x, t, pad, valid, DateWindowAligner(2, 0, 20))
    return module, x, output, captured


def test_linear_slope_real_dates_mean_and_observation_information():
    _, _, (out, valid, counts, stats), projection = run([2, 8, 20], [1, 4, 10])
    assert counts.flatten().tolist() == [2, 1]
    assert out.flatten().tolist() == [5, 20]
    assert valid.all()
    assert projection[0][0, 0, 0, 0] == 20  # 每日变化2，窗口宽10
    assert projection[1][0, 0, 0, 0] == 0
    assert stats["mean_dates"].flatten().tolist() == [2.5, 10.]
    torch.testing.assert_close(stats["info"][0, 0, :, 0, 0],
                               torch.tensor([-.25, torch.log(torch.tensor(3.)), .3, 1., 1.]))


@pytest.mark.parametrize("dates,expected", [([3, 3], False), ([3, 4], False), ([3, 5], True)])
def test_repeated_dates_and_minimum_span(dates, expected):
    _, _, (_, _, _, stats), projection = run([1, 2], dates)
    assert bool(stats["trend_valid"][0, 0]) == expected
    assert torch.isfinite(projection[0]).all()
    if not expected:
        assert projection[0][0, 0].count_nonzero() == 0


def test_invalid_nan_and_empty_windows_have_zero_features_and_metadata():
    mask = torch.tensor([[True, False]])
    _, x, (out, valid, _, stats), _ = run([7, float("nan")], [2, float("nan")], mask)
    assert torch.isfinite(out).all() and out.flatten().tolist() == [7, 0]
    assert valid.flatten().tolist() == [True, False]
    assert stats["info"][:, 1].count_nonzero() == 0
    out.sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.flatten().tolist() == [1, 0]


def test_zero_length_sequence():
    _, _, (out, valid, counts, stats), _ = run([], [])
    assert out.shape == (1, 2, 1, 1, 1)
    assert not valid.any() and not counts.any() and not stats["info"].any()


def test_left_closed_right_open_edges_and_rejection():
    _, _, (_, _, counts, _), _ = run([1, 2], [0, 10])
    assert counts.flatten().tolist() == [1, 1]
    with pytest.raises(ValueError, match="start_day,end_day"):
        run([1], [20])


def test_zero_residual_exactly_matches_original_aligner_and_two_step_gradients():
    module, x, (out, _, _, _), _ = run([1, 4], [1, 4])
    expected = DateWindowAligner(2, 0, 20)(x, torch.tensor([[1., 4.]]), torch.zeros(1, 2, dtype=torch.bool))[0]
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    optimizer = torch.optim.SGD(module.parameters(), lr=.1)
    out.square().sum().backward()
    assert module.gamma.grad.abs() > 0
    optimizer.step()
    optimizer.zero_grad()
    module(x, torch.tensor([[1., 4.]]), torch.zeros(1, 2, dtype=torch.bool), None,
           DateWindowAligner(2, 0, 20))[0].square().sum().backward()
    assert module.projection[0].weight.grad.abs().sum() > 0


def test_feature_accumulation_does_not_overflow_half():
    module = DynamicWindowRepresentation(1).half()
    x = torch.full((1, 4, 1, 1, 1), 40000., dtype=torch.float16)
    out = module(x, torch.tensor([[1., 3., 5., 7.]]), torch.zeros(1, 4, dtype=torch.bool),
                 None, DateWindowAligner(2, 0, 20))[0]
    assert torch.isfinite(out).all() and out[0, 0] == 40000


def test_pixelwise_missing_dates_have_independent_slopes_and_counts():
    module = DynamicWindowRepresentation(1)
    captured = []
    module.projection[0].register_forward_pre_hook(lambda _, inputs: captured.append(inputs[0]))
    values = torch.tensor([[[[[1., 10.]]], [[[3., float("nan")]]], [[[5., 18.]]]]])
    valid = torch.tensor([[[[[True, True]]], [[[True, False]]], [[[True, True]]]]])
    out, _, counts, _ = module(values, torch.tensor([[1., 3., 5.]]),
                                torch.zeros(1, 3, dtype=torch.bool), valid, DateWindowAligner(2, 0, 20))
    assert out[0, 0, 0, 0].tolist() == [3., 14.]
    assert counts[0, 0, 0, 0].tolist() == [3, 2]
    torch.testing.assert_close(captured[0][0, 0, 0], torch.tensor([10., 20.]))
    _, _, _, projection = run([8, 2], [1, 4])
    assert projection[0][0, 0, 0, 0] == -20
