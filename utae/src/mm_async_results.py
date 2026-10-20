"""完整混淆矩阵、沿用基线 Void 口径的指标及五折/多种子汇总。"""

import argparse
import json
from pathlib import Path

import numpy as np


def metric_report(matrix, ignore_index=19):
    raw = np.asarray(matrix, dtype=np.int64)
    cm = raw.copy()
    keep = np.ones(len(cm), dtype=bool)
    if ignore_index is not None:
        keep[ignore_index] = False
        cm[ignore_index, :] = 0
        cm[:, ignore_index] = 0
    tp, support, predictions = np.diag(cm), cm.sum(1), cm.sum(0)

    def divide(a, b):
        return np.divide(a, b, out=np.full(len(cm), np.nan), where=b > 0)

    values = {"IoU": divide(tp, support + predictions - tp),
              "Precision": divide(tp, predictions), "Recall": divide(tp, support),
              "F1": divide(2 * tp, support + predictions)}
    per_class = {}
    for i in range(len(cm)):
        per_class[str(i)] = {k: float(v[i]) if keep[i] and np.isfinite(v[i]) else None
                             for k, v in values.items()}
        per_class[str(i)].update(ignored=not bool(keep[i]),
                                 pixels=int(raw[i].sum()), predictions=int(raw[:, i].sum()))
    overall = {"OA": float(tp.sum() / cm.sum()) if cm.sum() else None}
    total_tp = int(tp.sum())
    total_fp = int(predictions.sum() - total_tp)
    total_fn = int(support.sum() - total_tp)
    for name, numerator, denominator in (
        ("micro_IoU", total_tp, total_tp + total_fp + total_fn),
        ("micro_Precision", total_tp, total_tp + total_fp),
        ("micro_Recall", total_tp, total_tp + total_fn),
        ("micro_F1", 2 * total_tp, 2 * total_tp + total_fp + total_fn),
    ):
        overall[name] = numerator / denominator if denominator else None
    for name, key in (("mIoU", "IoU"), ("macro_F1", "F1"),
                      ("macro_Precision", "Precision"), ("macro_Recall", "Recall")):
        selected = values[key][keep & np.isfinite(values[key])]
        overall[name] = float(selected.mean()) if len(selected) else None
    return {"units": "fraction", "void_policy": "baseline: exclude Void row and column for scoring",
            "confusion_matrix": raw.tolist(), "per_class": per_class, "overall": overall}


def summarize(root):
    root = Path(root)
    reports = [json.loads((root / f"Fold_{fold}" / "class_metrics.json").read_text()) for fold in range(1, 6)]
    keys = reports[0]["overall"]
    equal = {k: float(np.mean([r["overall"][k] for r in reports])) for k in keys
             if all(r["overall"][k] is not None for r in reports)}
    configs = [json.loads((root / f"Fold_{f}" / "config.json").read_text()) for f in range(1, 6)]
    normalized = [{k: v for k, v in cfg.items() if k not in {"fold", "N_params"}} for cfg in configs]
    if any(cfg != normalized[0] for cfg in normalized[1:]):
        raise ValueError("cannot aggregate folds with different configurations")
    combined = np.sum([r["confusion_matrix"] for r in reports], axis=0)
    result = {"fold_equal_mean": equal,
              "merged_confusion_metrics": metric_report(combined, configs[0]["ignore_index"])}
    (root / "async_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def summarize_seeds(roots):
    roots = [Path(root) for root in roots]
    configs = [json.loads((root / "Fold_1" / "config.json").read_text()) for root in roots]
    excluded = {"rdm_seed", "res_dir", "fold", "N_params"}
    protocols = [{k: v for k, v in cfg.items() if k not in excluded} for cfg in configs]
    if any(protocol != protocols[0] for protocol in protocols[1:]):
        raise ValueError("cannot aggregate seeds with different protocols")
    seeds = [cfg.get("rdm_seed") for cfg in configs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate seeds cannot be treated as independent runs")
    results = [summarize(root) for root in roots]

    def statistics(items):
        return {key: {"mean": float(np.mean([item[key] for item in items])),
                       "std": float(np.std([item[key] for item in items], ddof=1)) if len(items) > 1 else 0.}
                for key in items[0] if all(item.get(key) is not None for item in items)}

    result = {"seeds": seeds, "std_definition": "sample standard deviation (ddof=1)",
              "fold_equal_mean": statistics([r["fold_equal_mean"] for r in results]),
              "merged_confusion_metrics": statistics([r["merged_confusion_metrics"]["overall"] for r in results])}
    (roots[0].parent / "async_multiseed_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", help="one result directory per seed")
    args = parser.parse_args()
    result = summarize_seeds(args.roots) if len(args.roots) > 1 else summarize(args.roots[0])
    print(json.dumps(result, indent=2))
