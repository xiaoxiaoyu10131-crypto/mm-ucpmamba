"""真实 CPU U-TAE 分支的多模态融合、日期、mask 和梯度回归测试。"""
from io import BytesIO
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backbones.mm_utae import MMUTAE


@pytest.fixture(scope="module", autouse=True)
def _cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _model(**kwargs):
    options = dict(
        optical_input_dim=10,
        sar_input_dim=3,
        sar_orbits="both",
        encoder_widths=[16, 16, 16, 32],
        decoder_widths=[16, 16, 16, 32],
        out_conv=[16, 20],
        n_head=4,
        d_model=32,
        d_k=2,
        pad_value=0,
        head_mode="logits",
    )
    options.update(kwargs)
    return MMUTAE(**options)


@pytest.fixture
def inputs():
    generator = torch.Generator().manual_seed(37)
    result = {}
    for modality, steps, channels in (("s2", 2, 10), ("s1a", 3, 3), ("s1d", 4, 3)):
        result[f"x_{modality}"] = torch.randn(
            2, steps, channels, 32, 32, generator=generator
        )
        dates = torch.arange(steps, dtype=torch.float32) * 13 + 7
        result[f"batch_positions_{modality}"] = dates.expand(2, -1).clone()
    return result


def _assert_gradient(parameter):
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()
    assert parameter.grad.abs().sum().item() > 0


def _last_leaf(model):
    return [module for module in model.modules() if not list(module.children())][-1]


def test_independent_branches_forward_and_backward(inputs):
    torch.manual_seed(9)
    model = _model().train()
    branches = [model.s2_branch, model.s1a_branch, model.s1d_branch]
    parameter_sets = [{p.data_ptr() for p in branch.parameters()} for branch in branches]
    for index, pointers in enumerate(parameter_sets):
        assert pointers
        for other in parameter_sets[index + 1:]:
            assert pointers.isdisjoint(other)
    for branch in branches:
        assert branch.encoder is True
        assert isinstance(branch.out_conv, nn.Identity)
        assert branch.temporal_encoder.return_att is True
    prediction = model(**inputs)
    assert prediction.shape == (2, 20, 32, 32)
    assert torch.isfinite(prediction).all()
    target = torch.randint(0, 20, (2, 32, 32))
    F.cross_entropy(prediction, target).backward()
    for branch in branches:
        _assert_gradient(branch.temporal_encoder.attention_heads.Q)
        _assert_gradient(branch.temporal_encoder.attention_heads.fc1_k.weight)
        _assert_gradient(next(branch.in_conv.parameters()))
        _assert_gradient(next(branch.down_blocks[0].parameters()))
        _assert_gradient(next(branch.up_blocks[0].parameters()))


def test_appended_padding_does_not_change_eval_predictions(inputs):
    model = _model().eval()
    extended = {}
    for modality in ("s2", "s1a", "s1d"):
        value = inputs[f"x_{modality}"]
        extended[f"x_{modality}"] = torch.cat(
            (value, value.new_zeros(2, 2, *value.shape[2:])), dim=1
        )
        dates = inputs[f"batch_positions_{modality}"]
        extended[f"batch_positions_{modality}"] = torch.cat(
            (dates, dates.new_full((2, 2), 999)), dim=1
        )
    with torch.no_grad():
        expected, actual = model(**inputs), model(**extended)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("modality", ("s2", "s1a", "s1d"))
def test_each_branch_uses_its_own_dates(inputs, modality):
    torch.manual_seed(12)
    model = _model().eval()
    changed = dict(inputs)
    dates = inputs[f"batch_positions_{modality}"].clone()
    dates[:, -1] += 143
    changed[f"batch_positions_{modality}"] = dates
    with torch.no_grad():
        expected, actual = model(**inputs), model(**changed)
    assert (expected - actual).abs().max().item() > 1e-6


def test_auxiliary_heads_receive_gradients(inputs):
    model = _model(auxiliary=True).train()
    result = model(**inputs, return_aux=True)
    assert set(result) == {"logits", "logits_s2", "logits_s1a", "logits_s1d"}
    target = torch.randint(0, 20, (2, 32, 32))
    for prediction in result.values():
        assert prediction.shape == (2, 20, 32, 32)
        prediction.retain_grad()
    sum(F.cross_entropy(value, target) for value in result.values()).backward()
    for prediction in result.values():
        _assert_gradient(prediction)
    aux_parameters = [
        parameter for name, parameter in model.named_parameters() if "aux" in name
    ]
    assert aux_parameters, "辅助分类器参数应可从模型中识别"
    for parameter in aux_parameters:
        _assert_gradient(parameter)


@pytest.mark.parametrize("orbit, active, absent", (("S1A", "s1a", "s1d"), ("S1D", "s1d", "s1a")))
def test_single_sar_orbit(inputs, orbit, active, absent):
    model = _model(sar_orbits=orbit, auxiliary=True).eval()
    selected = {key: value for key, value in inputs.items() if absent not in key}
    with torch.no_grad():
        prediction = model(**selected)
        auxiliary = model(**selected, return_aux=True)
    assert prediction.shape == (2, 20, 32, 32)
    assert getattr(model, f"{absent}_branch") is None
    assert set(auxiliary) == {"logits", "logits_s2", f"logits_{active}"}


@pytest.mark.parametrize("modality", ("s2", "s1a", "s1d"))
def test_missing_dates_are_rejected(inputs, modality):
    changed = dict(inputs)
    changed[f"batch_positions_{modality}"] = None
    with pytest.raises(ValueError):
        _model()(**changed)


@pytest.mark.parametrize("modality", ("s2", "s1a", "s1d"))
def test_missing_modality_is_rejected(inputs, modality):
    changed = dict(inputs)
    changed[f"x_{modality}"] = None
    with pytest.raises(ValueError):
        _model()(**changed)


@pytest.mark.parametrize("modality", ("s2", "s1a", "s1d"))
def test_all_padding_sample_is_rejected(inputs, modality):
    changed = dict(inputs)
    value = inputs[f"x_{modality}"].clone()
    value[0].zero_()
    changed[f"x_{modality}"] = value
    with pytest.raises(ValueError):
        _model()(**changed)


@pytest.mark.parametrize("modality", ("s2", "s1a", "s1d"))
def test_wrong_date_shape_is_rejected(inputs, modality):
    changed = dict(inputs)
    changed[f"batch_positions_{modality}"] = torch.ones(2, 99)
    with pytest.raises(ValueError):
        _model()(**changed)


def test_checkpoint_round_trip(inputs):
    model = _model().eval()
    buffer = BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = _model().eval()
    restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
    with torch.no_grad():
        expected, actual = model(**inputs), restored(**inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_raw_logit_head_keeps_negative_class_scores(inputs):
    model = _model().eval()
    final = _last_leaf(model.fusion_head)
    assert isinstance(final, nn.Conv2d)
    assert final.out_channels == 20
    with torch.no_grad():
        final.weight.zero_()
        final.bias.fill_(-2)
        prediction = model(**inputs)
    torch.testing.assert_close(prediction, torch.full_like(prediction, -2), rtol=0, atol=0)


def test_legacy_head_retains_official_relu(inputs):
    model = _model(head_mode="legacy").eval()
    assert isinstance(_last_leaf(model.fusion_head), nn.ReLU)
    with torch.no_grad():
        prediction = model(**inputs)
    assert (prediction >= 0).all()

