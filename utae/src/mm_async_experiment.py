"""Async 独立入口、配置和 checkpoint 元数据。"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def add_async_arguments(parser, inference=False):
    defaults = dict(radius=1, dim=16, trend_min_span=2.0, residual_init=0.0)
    for key, default in defaults.items():
        parser.add_argument("--async-" + key.replace("_", "-"),
                            default=None if inference else default, type=type(default))
    for key in ("dynamic", "interaction"):
        parser.add_argument("--async-" + key, action=argparse.BooleanOptionalAction,
                            default=None if inference else True)


def async_settings(config):
    return {key: getattr(config, "async_" + key, default) for key, default in (
        ("radius", 1), ("dim", 16), ("trend_min_span", 2.0),
        ("dynamic", True), ("interaction", True), ("residual_init", 0.0),
    )}


def configure_async(config):
    if config.model != "mm_ucpmamba_async":
        return
    for key, value in (("mid_start_day", -200.), ("mid_end_day", 600.)):
        if getattr(config, key, None) is None:
            setattr(config, key, value)
    if (getattr(config, "sar_satellite", None) is not None
            or getattr(config, "sar_orbits", "both") != "both"
            or getattr(config, "optical_satellite", "S2") != "S2"):
        raise ValueError("async requires S2/S1A/S1D")
    if getattr(config, "fusion_mode", "concat") != "concat" or getattr(config, "use_quality_temporal", False):
        raise ValueError("async requires quality-free concat fusion")
    if getattr(config, "modality_dropout", 0.0) != 0:
        raise ValueError("first async experiment requires modality_dropout=0")
    config.async_design = async_settings(config)
    config.async_design["reference_date"] = config.ref_date


def build_async(config):
    from src.backbones.mm_ucpmamba_async import MMUCPMambaAsync

    configure_async(config)
    settings = async_settings(config)
    return MMUCPMambaAsync(
        optical_input_dim=getattr(config, "optical_input_dim", 10),
        sar_input_dim=getattr(config, "sar_input_dim", 3),
        encoder_widths=config.encoder_widths, decoder_widths=config.decoder_widths,
        out_conv=config.out_conv, str_conv_k=config.str_conv_k,
        str_conv_s=config.str_conv_s, str_conv_p=config.str_conv_p,
        encoder_norm=config.encoder_norm, decoder_norm=config.decoder_norm,
        decoder_groups=config.decoder_groups, hidden_dim=64,
        padding_mode=config.padding_mode, pad_value=config.pad_value,
        num_bins=config.mid_num_bins, start_day=config.mid_start_day, end_day=config.mid_end_day,
        low_memory=getattr(config, "mid_low_memory", False),
        async_radius=settings["radius"], async_dim=settings["dim"],
        trend_min_span=settings["trend_min_span"], dynamic_enabled=settings["dynamic"],
        interaction_enabled=settings["interaction"], residual_init=settings["residual_init"],
    )


def check_async_checkpoint(checkpoint, config):
    if config.model == "mm_ucpmamba_async":
        configure_async(config)
        saved = checkpoint.get("model_metadata", {}).get("async_design")
        if saved != config.async_design:
            raise ValueError("async checkpoint configuration/reference date mismatch")


def source_provenance():
    root = Path(__file__).resolve().parents[1]
    command = ["git", "-c", f"safe.directory={root.parent.as_posix()}"]
    try:
        revision = subprocess.check_output(command + ["rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(command + ["status", "--porcelain"], cwd=root, text=True))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    paths = sorted((root / "src").rglob("*.py")) + [root / name for name in (
        "train_semantic.py", "test_semantic.py", "run_mm_ucpmamba_async.py",
        "run_mm_ucpmamba_async.sh", "mm-ucpmamba-async.sh",
    )]
    hashes, missing, read_errors = {}, [], {}
    for path in paths:
        name = str(path.relative_to(root))
        try:
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        except FileNotFoundError:
            missing.append(name)
        except OSError as error:
            read_errors[name] = f"{type(error).__name__}: {error}"
    return {"git_revision": revision, "dirty": dirty, "source_sha256": hashes,
            "missing_files": missing, "unreadable_files": read_errors}


def save_async_last(model, optimizer, config, fold, epoch, best_miou):
    import torch

    target = Path(config.res_dir) / f"Fold_{fold}"
    torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(), "best_val_miou": best_miou,
                "config": vars(config), "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "model_metadata": {"model_name": config.model, "async_design": config.async_design}},
               target / "last.pth.tar")


def save_async_metrics(target, matrix, ignore_index):
    from src.mm_async_results import metric_report

    Path(target).write_text(json.dumps(metric_report(matrix, ignore_index), indent=2,
                                      allow_nan=False), encoding="utf-8")


def benchmark_async(model, images, dates, forward, config):
    """同一实际测试 batch：预热一次、同步计时三次；不计数据加载。"""
    import torch

    device = next(model.parameters()).device
    with torch.no_grad():
        forward(model, images, dates, config)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(3):
            forward(model, images, dates, config)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return {"ms_per_batch": (time.perf_counter() - start) * 1000 / 3,
            "warmup": 1, "repeats": 3, "data_loading_included": False,
            "input_shapes": {n: list(x.shape) for n, x in images.items() if torch.is_tensor(x)}}
