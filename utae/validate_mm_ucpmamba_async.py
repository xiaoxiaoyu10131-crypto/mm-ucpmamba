"""真实 PASTIS-R + 真实 Mamba CUDA 的更新、恢复和运行成本检查。"""

import argparse
import json
from pathlib import Path
import tempfile
import time

import torch

from run_mm_ucpmamba_async import make_config
from src import model_utils
from src.dataset import PASTIS_Dataset
from src.folds import get_fold_split
from src.learning.weight_init import reset_final_batchnorm_identity, weight_init
from src.mm_async_collate import async_pad_collate
from src.mm_async_experiment import configure_async, source_provenance
from train_semantic import _forward_model, recursive_todevice


def run(args, report):
    if args.batch_size < 1 or args.steps < 2 or args.patches < args.batch_size:
        raise ValueError("batch_size>=1, steps>=2, patches>=batch_size required")
    if not torch.cuda.is_available():
        raise RuntimeError("this validation requires real CUDA/Mamba")
    torch.manual_seed(1)
    torch.set_num_threads(2)
    device = torch.device("cuda")
    config = make_config(["--dataset_folder", args.dataset_folder, "--batch_size", str(args.batch_size)])
    configure_async(config)
    folds, _, _ = get_fold_split(args.fold)
    dataset = PASTIS_Dataset(folder=args.dataset_folder, norm=True, target="semantic",
                             sats=["S2", "S1A", "S1D"], reference_date=config.ref_date,
                             folds=folds, temporal_dropout=None)
    # 按真实长度选长序列；不加载测试折，也不截断帧。
    lengths = {int(pid): {m: len(dataset.get_dates(pid, m)) for m in dataset.sats}
               for pid in dataset.id_patches}
    ids = (sorted(lengths, key=lambda pid: sum(lengths[pid].values()), reverse=True)
           if args.selection == "longest" else list(lengths))[:args.patches]
    index = {int(pid): i for i, pid in enumerate(dataset.id_patches)}
    selected = [dataset[index[pid]] for pid in ids]
    model = model_utils.get_model(config).to(device)
    model.apply(weight_init)
    reset_final_batchnorm_identity(model, config.num_classes)
    model.reset_async_initialization()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    weights = torch.ones(config.num_classes, device=device)
    weights[config.ignore_index] = 0
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    report.update(config=vars(config), source=source_provenance(),
                  parameters=sum(p.numel() for p in model.parameters()),
                  gpu=torch.cuda.get_device_name(device), torch_version=torch.__version__,
                  physical_batch=args.batch_size, precision="float32", training_folds=folds,
                  patch_ids=ids, full_lengths={pid: lengths[pid] for pid in ids},
                  selection=args.selection, full_sequences=True, temporal_dropout=False, losses=[])
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    started = time.perf_counter()
    for step in range(args.steps):
        start = (step * args.batch_size) % len(selected)
        samples = [selected[(start + i) % len(selected)] for i in range(args.batch_size)]
        (images, dates), target = recursive_todevice(async_pad_collate(samples), device)
        optimizer.zero_grad()
        output = _forward_model(model, images, dates, config)
        loss = criterion(output, target.long())
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss")
        loss.backward()
        if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise RuntimeError("non-finite gradient")
        if step == args.steps - 1:
            report["gradient_norms"] = {}
            for prefix in ("spatial_branches", "dynamic_windows", "async_interaction", "temporal_encoder", "up_blocks"):
                grads = [p.grad for n, p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
                norm = sum(float(g.abs().sum()) for g in grads)
                report["gradient_norms"][prefix] = norm
                if norm <= 0:
                    raise RuntimeError(f"missing trainable gradient: {prefix}")
        optimizer.step()
        report["losses"].append(float(loss.detach()))
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"step {step + 1}/{args.steps}: loss={loss.item():.6f}", flush=True)
        del output, loss
    torch.cuda.synchronize(device)
    report["training_seconds"] = time.perf_counter() - started
    report["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
    report["peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2**20
    report["last_batch_shapes"] = {n: list(x.shape) for n, x in images.items() if torch.is_tensor(x)}
    report["module_diagnostics"] = {k: v.cpu().tolist() for k, v in model.last_diagnostics.items()}
    cycle = (len(selected) + args.batch_size - 1) // args.batch_size
    report["training_cycle_mean_losses"] = [sum(report["losses"][i:i + cycle]) / cycle
                                            for i in range(0, args.steps - cycle + 1, cycle)]
    report["small_sample_check"] = 8 <= len(selected) <= 16 and args.steps >= cycle * 2
    model.eval()
    with torch.no_grad():
        expected = _forward_model(model, images, dates, config)
        with tempfile.TemporaryDirectory(dir=Path(args.output).parent) as folder:
            checkpoint = Path(folder) / "checkpoint.pt"
            torch.save(model.state_dict(), checkpoint)
            restored = model_utils.get_model(config).to(device).eval()
            restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            report["checkpoint_state_exact_match"] = all(
                torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items()
            )
            if not report["checkpoint_state_exact_match"]:
                raise RuntimeError("checkpoint tensors changed during restoration")
            actual = _forward_model(restored, images, dates, config)
            report["checkpoint_output_max_abs_error"] = float((actual - expected).abs().max())
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
            del actual, restored
        report["checkpoint_output_close"] = True
        report["checkpoint_output_tolerance"] = {"rtol": 1e-5, "atol": 1e-5}
        report["output_shape"] = list(expected.shape)
        _forward_model(model, images, dates, config)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(3):
            _forward_model(model, images, dates, config)
        torch.cuda.synchronize(device)
        report["inference_ms_per_batch"] = (time.perf_counter() - start) * 1000 / 3
    report["status"] = "passed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-folder", required=True)
    parser.add_argument("--fold", type=int, default=2, choices=range(1, 6))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patches", type=int, default=8, help="8–16 for small-sample fitting")
    parser.add_argument("--steps", type=int, default=2, help="increase for a separate small-sample fitting check")
    parser.add_argument("--selection", choices=["longest", "first"], default="longest")
    parser.add_argument("--output", default="outputs/mm_ucpmamba_async_validation.json")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report: dict = {"purpose": "real-data implementation check; not research performance", "status": "running"}
    try:
        run(args, report)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        if torch.cuda.is_available():
            report["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        raise
    finally:
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
