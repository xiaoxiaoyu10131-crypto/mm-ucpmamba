"""实际 collate、训练前向入口与推理配置恢复的 CPU 测试。"""

import argparse
import ast
import importlib
import os
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_mm_ucpmamba_mid import mid_module as _mid_fixture, model


mid_module = _mid_fixture
ROOT = Path(__file__).resolve().parents[1]


def sample(lengths, include_masks=True):
    data, dates = {}, {}
    for name, length, channels in zip(("S2", "S1A", "S1D"), lengths, (10, 3, 3)):
        data[name] = torch.randn(length, channels, 16, 16)
        dates[name] = torch.linspace(1, 29, length)
    if include_masks:
        data["pad_masks"] = {name: torch.zeros(length, dtype=torch.bool)
                             for name, length in zip(("S2", "S1A", "S1D"), lengths)}
        data["valid_masks"] = {name: torch.ones(length, 1, 16, 16, dtype=torch.bool)
                               for name, length in zip(("S2", "S1A", "S1D"), lengths)}
    return ((data, dates), torch.zeros(16, 16, dtype=torch.long))


def train_helpers():
    tree = ast.parse((ROOT / "train_semantic.py").read_text(encoding="utf-8"))
    names = {"_active_sar_orbits", "_mm_satellites", "_is_mm_model",
             "_set_mm_experiment_metadata", "_forward_model"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"os": os}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "train_semantic.py", "exec"), namespace)
    return namespace


def config(name="mm_ucpmamba_mid"):
    return SimpleNamespace(model=name, optical_satellite="S2", sar_orbits="both",
                           res_dir="./results", ref_date="2018-09-01", mid_num_bins=24,
                           mid_start_day=0, mid_end_day=40)


def inference_parser():
    tree = ast.parse((ROOT / "test_semantic.py").read_text(encoding="utf-8"))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "parser"
                                                for t in node.targets):
            nodes.append(node)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "parser":
                nodes.append(node)
    namespace = {"argparse": argparse}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "test_semantic.py", "exec"), namespace)
    return namespace["parser"], tree


def test_collate_preserves_real_zero_internal_padding_and_masks(mid_module):
    collate = importlib.import_module("src.mm_collate").mid_pad_collate
    first, second = sample((2, 3, 1)), sample((4, 1, 2))
    first[0][0]["S2"][0].zero_()
    first[0][0]["pad_masks"]["S2"][1] = True
    first[0][0]["valid_masks"]["S2"][0, :, 0, 0] = False
    (data, dates), target = collate([first, second])
    assert data["S2"].shape == (2, 4, 10, 16, 16)
    assert data["S1A"].shape[1] == 3 and data["S1D"].shape[1] == 2
    assert data["pad_masks"]["S2"].tolist() == [[False, True, True, True], [False] * 4]
    assert data["pad_masks"]["S1A"].tolist() == [[False] * 3, [False, True, True]]
    assert data["pad_masks"]["S1D"].tolist() == [[False, True], [False, False]]
    assert data["S2"][0, 0].count_nonzero() == 0
    assert not data["valid_masks"]["S2"][0, 2:].any()
    assert not data["valid_masks"]["S2"][0, 0, :, 0, 0].any()
    assert data["valid_masks"]["S2"].dtype == torch.bool
    assert dates["S2"].shape == (2, 4) and target.shape == (2, 16, 16)


def test_collate_without_user_masks_still_constructs_explicit_padding(mid_module):
    collate = importlib.import_module("src.mm_collate").mid_pad_collate
    (data, _), _ = collate([sample((2, 3, 1), False), sample((4, 1, 2), False)])
    assert data["pad_masks"]["S2"].tolist() == [[False, False, True, True], [False] * 4]


def test_collate_optional_masks_can_be_absent_in_one_sample(mid_module):
    collate = importlib.import_module("src.mm_collate").mid_pad_collate
    first, second = sample((2, 3, 1)), sample((4, 1, 2), False)
    first[0][0]["pad_masks"]["S2"][1] = True
    (data, _), _ = collate([first, second])
    assert data["pad_masks"]["S2"].tolist() == [[False, True, True, True], [False] * 4]
    assert data["valid_masks"]["S2"][1].all()


def test_collate_mixes_frame_pixel_masks_and_different_optional_modality_keys(mid_module):
    collate = importlib.import_module("src.mm_collate").mid_pad_collate
    first, second = sample((2, 3, 1), False), sample((4, 1, 2), False)
    first[0][0]["valid_masks"] = {"S2": torch.tensor([True, False])}
    first[0][0]["pad_masks"] = {"S1A": torch.tensor([False, True, False])}
    second[0][0]["valid_masks"] = {"S1D": torch.ones(2, 1, 16, 16, dtype=torch.bool)}
    second[0][0]["valid_masks"]["S1D"][1, :, 0, 0] = False
    (data, _), _ = collate([first, second])
    assert data["valid_masks"]["S2"].shape == (2, 4, 1, 16, 16)
    assert not data["valid_masks"]["S2"][0, 1:].any()
    assert data["valid_masks"]["S2"][1].all()
    assert not data["valid_masks"]["S1A"][0, 1].any()
    assert not data["valid_masks"]["S1D"][1, 1, :, 0, 0].any()
    assert set(first[0][0]["valid_masks"]) == {"S2"}
    assert first[0][0]["valid_masks"]["S2"].shape == (2,)
    assert "pad_masks" not in second[0][0]


def test_collate_rejects_mismatched_date_length(mid_module):
    collate = importlib.import_module("src.mm_collate").mid_pad_collate
    item = sample((2, 3, 1))
    item[0][1]["S2"] = item[0][1]["S2"][:1]
    with pytest.raises(ValueError, match="different lengths"):
        collate([item])


def test_training_forward_only_passes_masks_to_new_interface(mid_module):
    forward = train_helpers()["_forward_model"]
    collate = importlib.import_module("src.mm_collate").mid_pad_collate
    (data, dates), _ = collate([sample((2, 3, 1)), sample((4, 1, 2))])
    calls = []

    def recorder(*args, **kwargs):
        calls.append((args, kwargs))
        return "prediction"

    assert forward(recorder, data, dates, config()) == "prediction"
    assert calls[-1][1]["pad_masks"] is data["pad_masks"]
    assert calls[-1][1]["valid_masks"] is data["valid_masks"]
    assert calls[-1][1]["batch_positions_s1a"] is dates["S1A"]
    assert forward(recorder, data, dates, config("mm_ucpmamba")) == "prediction"
    assert set(calls[-1][1]) == {"batch_positions_s2", "batch_positions_s1a", "batch_positions_s1d"}
    assert all(arg is data[name] for arg, name in zip(calls[-1][0], ("S2", "S1A", "S1D")))


def test_real_collate_to_training_forward_to_mid_model(mid_module):
    collate = importlib.import_module("src.mm_collate").mid_pad_collate
    first, second = sample((2, 3, 1)), sample((4, 1, 2))
    first[0][0]["S2"][0].zero_()
    (data, dates), _ = collate([first, second])
    instance = model(mid_module).eval()
    with torch.no_grad():
        logits = train_helpers()["_forward_model"](instance, data, dates, config())
        debug = instance(data["S2"], data["S1A"], data["S1D"],
                         batch_positions_s2=dates["S2"], batch_positions_s1a=dates["S1A"],
                         batch_positions_s1d=dates["S1D"], pad_masks=data["pad_masks"],
                         valid_masks=data["valid_masks"], return_debug=True)
    assert logits.shape == (2, 4, 16, 16) and torch.isfinite(logits).all()
    assert not debug["shallow_pad_masks"]["S2"][0, 0]
    torch.testing.assert_close(logits, debug["logits"], rtol=0, atol=0)


def test_default_experiment_names_and_calendar_metadata_are_independent():
    metadata = train_helpers()["_set_mm_experiment_metadata"]
    original, new = config("mm_ucpmamba"), config()
    metadata(original)
    metadata(new)
    assert original.experiment_name == "mm_ucpmamba_s2_s1a_s1d"
    assert new.experiment_name == "mm_ucpmamba_mid_s2_s1a_s1d"
    assert original.res_dir != new.res_dir
    assert not hasattr(original, "mid_alignment")
    assert new.mid_alignment == dict(
        num_bins=24, start_day=0, end_day=40,
        reference_date="2018-09-01", bounds="[start_day,end_day)",
    )


def test_inference_none_defaults_preserve_saved_calendar_and_explicit_override():
    parser, tree = inference_parser()
    main = next(node for node in tree.body if isinstance(node, ast.If))
    cli_expr = next(node.value for node in main.body if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "cli_config" for t in node.targets))
    merge_expr = next(node.value for node in main.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "config" for t in node.targets)
                      and isinstance(node.value, ast.Dict))
    saved = dict(model="mm_ucpmamba_mid", mid_num_bins=24, mid_start_day=-20, mid_end_day=380)
    cli_code = compile(ast.Expression(cli_expr), "test_semantic.py", "eval")
    merge_code = compile(ast.Expression(merge_expr), "test_semantic.py", "eval")
    cases = [
        ([], saved),
        (["--mid-num-bins", "12", "--mid-end-day", "400"],
         dict(saved, mid_num_bins=12, mid_end_day=400)),
    ]
    for arguments, expected in cases:
        parsed = parser.parse_args(arguments)
        cli = eval(cli_code, {"test_config": parsed})
        restored = eval(merge_code, {"model_config": saved, "cli_config": cli})
        for key in saved:
            assert restored[key] == expected[key]


@pytest.mark.parametrize("filename", ["train_semantic.py", "test_semantic.py"])
def test_collate_selection_uses_mid_only(filename):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    guards = [node for node in ast.walk(tree) if isinstance(node, ast.If)
              and "mid_pad_collate" in ast.unparse(node)]
    guard = next(node for node in guards if isinstance(node.test, ast.Compare)
                 and ast.unparse(node.test) == "config.model == 'mm_ucpmamba_mid'")
    selected = next(node for node in guard.body if isinstance(node, ast.Assign))
    code = compile(ast.Module(body=[guard], type_ignores=[]), filename, "exec")
    marker = object()
    for name, changes in [("mm_ucpmamba_mid", True), ("mm_ucpmamba", False)]:
        namespace = {"config": SimpleNamespace(model=name, pad_value=0), "collate_fn": marker,
                     "partial": partial, "mid_pad_collate": lambda batch, pad_value: batch}
        exec(code, namespace)
        assert (namespace["collate_fn"] is not marker) == changes
    assert selected.targets[0].id == "collate_fn"
