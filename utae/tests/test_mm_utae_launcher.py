"""启动脚本的参数与保护测试；临时执行替身仅用于测试，不运行训练。"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(sys.platform == "win32" or not BASH,
                                reason="launcher tests require Linux/WSL bash")


@pytest.fixture
def launcher(tmp_path):
    dataset = tmp_path / "data"
    dataset.mkdir()
    calls = tmp_path / "calls.jsonl"
    fake = tmp_path / "test_python"
    fake.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ["TEST_CALLS"], "a") as handle:
    handle.write(json.dumps(args) + "\\n")
if args[0] == "-":
    if len(args) > 1 and args[1].endswith("conf.json"):
        sys.path.insert(0, os.getcwd())
        sys.argv = args
        exec(compile(sys.stdin.read(), "run_mm_utae.sh", "exec"))
    raise SystemExit(0)
fold = args[args.index("--fold") + 1]
if fold == os.environ.get("TEST_FAIL_FOLD"):
    raise SystemExit(7)
result = Path(args[args.index("--res_dir") + 1]) / f"Fold_{fold}"
result.mkdir(parents=True, exist_ok=True)
(result / "model.pth.tar").write_text("test-only checkpoint marker")
''', encoding="utf-8")
    fake.chmod(0o755)
    env = {**os.environ, "PYTHON": str(fake), "DATASET_FOLDER": str(dataset),
           "RES_DIR": str(tmp_path / "results"), "FOLDS": "1",
           "PROFILE": "comparison", "MODEL": "mm_utae", "TEST_CALLS": str(calls)}

    def run(**overrides):
        assert BASH is not None
        result = subprocess.run([BASH, str(ROOT / "run_mm_utae.sh")],
                                env={**env, **overrides}, capture_output=True, text=True,
                                timeout=60)
        recorded = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
        return result, recorded

    return run, env


@pytest.mark.parametrize("profile,model,head,weight", [
    ("comparison", "mm_utae", "legacy", "0"),
    ("paper", "mm_utae", "logits", "0.5"),
    ("comparison", "mm_ucpmamba", None, None),
])
def test_launcher_builds_explicit_model_protocol(launcher, profile, model, head, weight):
    run, _ = launcher
    result, calls = run(PROFILE=profile, MODEL=model)
    assert result.returncode == 0, result.stderr
    args = calls[0]
    assert args[args.index("--model") + 1] == model
    assert args[args.index("--ignore_index") + 1] == "19"
    assert args[args.index("--sar-orbits") + 1] == "both"
    if head is None:
        assert "--mm-utae-head" not in args and "--mm-utae-aux-weight" not in args
    else:
        assert args[args.index("--mm-utae-head") + 1] == head
        assert args[args.index("--mm-utae-aux-weight") + 1] == weight


@pytest.mark.parametrize("options", [
    {"PROFILE": "paper", "MODEL": "mm_ucpmamba"},
    {"FOLDS": "1 1"}, {"FOLDS": "6"}, {"BATCH_SIZE": "0"},
    {"DATASET_FOLDER": "/nonexistent/test_only_data"},
])
def test_invalid_launch_settings_do_not_start_training(launcher, options):
    run, _ = launcher
    result, calls = run(**options)
    assert result.returncode == 2 and calls == []


def test_existing_checkpoint_is_protected(launcher):
    run, env = launcher
    checkpoint = Path(env["RES_DIR"]) / "Fold_1" / "model.pth.tar"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("existing")
    result, calls = run()
    assert result.returncode == 2 and calls == []
    assert checkpoint.read_text() == "existing"


def test_only_five_folds_trigger_aggregation(launcher):
    run, _ = launcher
    result, calls = run(FOLDS="1 2 3 4 5")
    assert result.returncode == 0
    assert len(calls) == 6 and calls[-1][0] == "-"
    assert [args[args.index("--fold") + 1] for args in calls[:-1]] == ["1", "2", "3", "4", "5"]


def test_training_failure_stops_later_folds(launcher):
    run, _ = launcher
    result, calls = run(FOLDS="1 2 3", TEST_FAIL_FOLD="2")
    assert result.returncode == 7
    assert [args[args.index("--fold") + 1] for args in calls] == ["1", "2"]


def test_existing_configuration_must_match_before_new_fold(launcher):
    pytest.importorskip("torchnet")
    pytest.importorskip("geopandas")
    from train_semantic import parser

    run, env = launcher
    saved = vars(parser.parse_args(["--model", "mm_utae", "--ignore_index", "19",
                                   "--dataset_folder", env["DATASET_FOLDER"],
                                   "--final-bn-init", "ones", "--batch_size", "2"]))
    for key in ("encoder_widths", "decoder_widths", "out_conv"):
        saved[key] = json.loads(saved[key])
    result_dir = Path(env["RES_DIR"])
    result_dir.mkdir()
    (result_dir / "conf.json").write_text(json.dumps(saved))
    result, calls = run()
    assert result.returncode != 0
    assert "batch_size" in result.stderr
    assert len(calls) == 1 and calls[0][0] == "-"
    compatible, compatible_calls = run(BATCH_SIZE="2")
    assert compatible.returncode == 0, compatible.stderr
    assert compatible_calls[-1][0] == "-u"
