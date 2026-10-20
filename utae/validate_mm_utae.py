"""用真实 PASTIS-R 检查默认 MM-UTAE；一次更新不代表正式训练结果。"""

import argparse
import json
from pathlib import Path

import torch

from src import model_utils, utils
from src.dataset import PASTIS_Dataset
from src.folds import get_fold_split
from src.learning.weight_init import weight_init
from src.mm_utae_experiment import initialize_mm_utae_heads, mm_utae_loss
from train_semantic import _forward_model, parser as training_parser, recursive_todevice


def sample_batch(dataset, max_frames=0):
    (images, dates), target = dataset[0]
    original_lengths = {key: len(value) for key, value in images.items()}
    if max_frames > 0:
        for key, value in images.items():
            if len(value) > max_frames:
                indices = torch.linspace(0, len(value) - 1, max_frames).long()
                images[key], dates[key] = value[indices], dates[key][indices]
    return utils.pad_collate([((images, dates), target)]), original_lengths


def main(args):
    if args.max_frames < 0 or args.aux_weight < 0:
        raise ValueError("max_frames and aux_weight must be non-negative")
    torch.manual_seed(17)
    torch.set_num_threads(2)
    cfg = training_parser.parse_args([
        "--model", "mm_utae", "--device", args.device,
        "--mm-utae-head", args.head, "--mm-utae-aux-weight", str(args.aux_weight),
        "--ignore_index", "19", "--final-bn-init",
        "ones" if args.head == "legacy" else "legacy",
    ])
    for name in ("encoder_widths", "decoder_widths", "out_conv"):
        setattr(cfg, name, json.loads(getattr(cfg, name)))
    train_folds, _, test_folds = get_fold_split(args.fold)
    common = dict(folder=args.dataset_folder, norm=True, target="semantic",
                  sats=["S2", "S1A", "S1D"], reference_date=cfg.ref_date)
    training = PASTIS_Dataset(**common, folds=train_folds,
                             temporal_dropout={"S2": 0.4, "S1A": 0.2, "S1D": 0.2})
    testing = PASTIS_Dataset(**common, folds=test_folds)
    train_batch, training_lengths = sample_batch(training, args.max_frames)
    test_batch, testing_lengths = sample_batch(testing, args.max_frames)
    device = torch.device(args.device)
    model = model_utils.get_model(cfg).to(device)
    model.apply(weight_init)
    if args.head == "legacy":
        initialize_mm_utae_heads(model, cfg.num_classes)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=19)
    report = {
        "purpose": "real-data smoke validation; not benchmark performance",
        "dataset_folder": args.dataset_folder,
        "total_patches": len(testing.date_tables["S2"]),
        "official_test_fold": args.fold,
        "training_folds": train_folds,
        "train_patch_id": int(training.id_patches[0]),
        "test_patch_id": int(testing.id_patches[0]),
        "training_lengths_after_dropout_before_optional_subsample": training_lengths,
        "testing_full_lengths": testing_lengths,
        "max_frames": args.max_frames,
        "full_sequences_tested": args.max_frames == 0,
        "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "head": args.head, "auxiliary_weight": args.aux_weight,
        "physical_batch": 1, "torch_version": torch.__version__, "device": str(device),
    }
    if device.type == "cuda":
        report["gpu"] = torch.cuda.get_device_name(device)
        torch.cuda.reset_peak_memory_stats(device)
    model.eval()
    (images, dates), _ = recursive_todevice(test_batch, device)
    with torch.no_grad():
        prediction = _forward_model(model, images, dates, cfg)
    report["eval_output_shape"] = list(prediction.shape)
    report["eval_output_finite"] = bool(torch.isfinite(prediction).all())
    if not report["eval_output_finite"]:
        raise RuntimeError("non-finite evaluation output")
    del prediction, images, dates
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    (images, dates), target = recursive_todevice(train_batch, device)
    report["training_input_shapes"] = {key: list(value.shape) for key, value in images.items()}
    result = _forward_model(model, images, dates, cfg, return_aux=args.aux_weight > 0)
    prediction, loss = mm_utae_loss(result, target.long(), criterion, args.aux_weight)
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite training loss")
    loss.backward()
    report["loss"] = float(loss.detach())
    report["gradients_finite"] = all(bool(torch.isfinite(p.grad).all())
                                     for p in model.parameters() if p.grad is not None)
    report["attention_query_gradient_norms"] = {
        name: float(getattr(model, f"{name.lower()}_branch").temporal_encoder.attention_heads.Q.grad.norm())
        for name in model.active_modalities
    }
    if not report["gradients_finite"]:
        raise RuntimeError("non-finite gradients")
    parameter = next(model.fusion_head.parameters())
    before = parameter.detach().clone()
    optimizer.step()
    report["fusion_weight_update_max_abs"] = float((parameter.detach() - before).abs().max())
    if report["fusion_weight_update_max_abs"] <= 0:
        raise RuntimeError("optimizer did not update the fusion head")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        report["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-folder", required=True)
    parser.add_argument("--fold", default=2, type=int, choices=range(1, 6))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--head", choices=["legacy", "logits"], default="legacy")
    parser.add_argument("--aux-weight", default=0.0, type=float)
    parser.add_argument("--max-frames", default=0, type=int,
                        help="0 keeps full sequences; positive values explicitly limit smoke-test frames")
    parser.add_argument("--output", default="outputs/mm_utae_validation.json")
    main(parser.parse_args())
