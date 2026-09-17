"""
Main script for semantic experiments
Author: Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""
import argparse
import json
import os
import pickle as pkl
import pprint
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchnet as tnt
# import torchtnt as tnt
from src import utils, model_utils
from src.dataset import PASTIS_Dataset
from src.folds import FOLD_SPLITS, get_fold_split
from src.learning.metrics import confusion_matrix_analysis
from src.learning.miou import IoU
from src.learning.gradient_accumulation import (
    accumulation_window_size,
    is_accumulation_boundary,
)
from src.learning.weight_init import weight_init

parser = argparse.ArgumentParser()
# Model parameters
parser.add_argument(
    "--model",
    default="utae",
    type=str,
    choices=[
        "utae",
        "unet3d",
        "fpn",
        "convlstm",
        "convgru",
        "uconvlstm",
        "ucpmamba",
        "ucpmamba_vmambav2",
        "ucpmamba_mamba2",
        "ucpmamba_directvss",
        "ucpmamba_mean",
        "ucpmamba_plainlstm",
        "mm_ucpmamba",
        "buconvlstm",
    ],
    help=(
        "Semantic segmentation architecture. Use uconvlstm for the original "
        "baseline, ucpmamba, ucpmamba_vmambav2, ucpmamba_mamba2, and "
        "ucpmamba_mean, or mm_ucpmamba for independent Optical-SAR fusion."
    ),
)
## U-TAE Hyperparameters
parser.add_argument("--encoder_widths", default="[64,64,64,128]", type=str)
# 指定编码器的每层通道数，格式为字符串表示的列表
parser.add_argument("--decoder_widths", default="[32,32,64,128]", type=str)
# 指定解码器的每层通道数
parser.add_argument("--out_conv", default="[32, 20]")
# 指定输出卷积的层配置
parser.add_argument("--optical-input-dim", dest="optical_input_dim", default=10, type=int)
parser.add_argument("--sar-input-dim", dest="sar_input_dim", default=3, type=int)
parser.add_argument(
    "--fusion-mode", dest="fusion_mode", choices=["concat", "qaf"], default="concat"
)
complementary_group = parser.add_mutually_exclusive_group()
complementary_group.add_argument(
    "--use-complementary-residual",
    dest="use_complementary_residual",
    action="store_true",
)
complementary_group.add_argument(
    "--no-complementary-residual",
    dest="use_complementary_residual",
    action="store_false",
)
parser.set_defaults(use_complementary_residual=True)
parser.add_argument("--use-quality-temporal", dest="use_quality_temporal", action="store_true")
parser.add_argument("--quality-lambda", dest="quality_lambda", default=1.0, type=float)
parser.add_argument("--modality-dropout", dest="modality_dropout", default=0.0, type=float)
parser.add_argument("--temporal-dropout-s2", default=0.4, type=float)
parser.add_argument("--temporal-dropout-s1a", default=0.2, type=float)
parser.add_argument("--temporal-dropout-s1d", default=0.2, type=float)
parser.add_argument("--optical-satellite", dest="optical_satellite", default="S2", type=str)
parser.add_argument(
    "--sar-orbits",
    dest="sar_orbits",
    choices=["both", "S1A", "S1D"],
    default="both",
    help="Active SAR orbit(s); both is the formal tri-modal default.",
)
parser.add_argument(
    "--sar-satellite",
    dest="sar_satellite",
    choices=["S1A", "S1D"],
    default=None,
    help="Backward-compatible alias selecting one SAR orbit.",
)
parser.add_argument("--str_conv_k", default=4, type=int)
# 表示时空卷积的核大小
parser.add_argument("--str_conv_s", default=2, type=int)
# 定义卷积步幅
parser.add_argument("--str_conv_p", default=1, type=int)
# 表示卷积的填充大小
parser.add_argument("--agg_mode", default="att_group", type=str)
parser.add_argument("--encoder_norm", default="group", type=str)
# 定义编码器的归一化类型
parser.add_argument("--n_head", default=16, type=int)
# 定义多头注意力的头数量
parser.add_argument("--d_model", default=256, type=int)
# 模型的维度
parser.add_argument("--d_k", default=4, type=int)
# 每个注意力头的维度

# 数据集与训练配置
parser.add_argument(
    "--dataset_folder",
    default="",
    type=str,
    help="Path to the folder where the results are saved.",
)
# 数据集文件夹的路径
parser.add_argument(
    "--res_dir",
    default="./results",
    help="Path to the folder where the results should be stored",
)
# 用于存储结果的文件夹路径
parser.add_argument(
    "--num_workers", default=8, type=int, help="Number of data loading workers"
)
# 数据加载的并行工作线程数
parser.add_argument("--rdm_seed", default=1, type=int, help="Random seed")
parser.add_argument(
    "--device",
    default="cuda",
    type=str,
    help="Name of device to use for tensor computations (cuda/cpu)",
)
# 指定用于张量计算的设备，默认值为cuda
parser.add_argument(
    "--display_step",
    default=50,
    type=int,
    help="Interval in batches between display of training metrics",
)
# 每50次展示训练矩阵
parser.add_argument(
    "--cache",
    dest="cache",
    action="store_true",
    help="If specified, the whole dataset is kept in RAM",
)
# 是否将整个数据集保存在RAM中以加速数据加载。使用store_true表示只需指定此标志，而不需要提供值。
# Training parameters
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs per fold")
# 每折交叉验证的训练轮数
parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
parser.add_argument(
    "--accumulation_steps",
    default=1,
    type=int,
    help="Number of micro-batches accumulated before each optimizer update",
)
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
parser.add_argument("--mono_date", default=None, type=str)
# 单一日期模式的配置
parser.add_argument("--ref_date", default="2018-09-01", type=str)
# 参考日期，用于对时间序列进行标准化
parser.add_argument(
    "--fold",
    default=None,
    type=int,
    choices=range(1, 6),
    help="Train one official PASTIS test Fold (between 1 and 5)",
)
parser.add_argument("--num_classes", default=20, type=int)
# 分类任务的类别数量
parser.add_argument("--ignore_index", default=-1, type=int)
# 指定在损失计算时忽略的索引
parser.add_argument("--pad_value", default=0, type=float)
# 数据填充时的填充值
parser.add_argument("--padding_mode", default="reflect", type=str)
parser.add_argument(
    "--val_every",
    default=1,
    type=int,
    help="Interval in epochs between two validation steps.",
)
parser.add_argument(
    "--val_after",
    default=0,
    type=int,
    help="Do validation only after that many epochs.",
)
parser.add_argument(
    "--diagnostics",
    action="store_true",
    help="Record per-epoch diagnostics for one semantic class.",
)
parser.add_argument(
    "--diagnostic_class",
    default=13,
    type=int,
    help="PASTIS semantic class index to diagnose (13 is Potatoes).",
)

list_args = ["encoder_widths", "decoder_widths", "out_conv"]
# 存储需要特殊处理的参数名
parser.set_defaults(cache=False)
# 为cache参数设置默认值为False


PASTIS_LABEL_NAMES = {
    0: "Background", 1: "Meadow", 2: "Soft winter wheat", 3: "Corn",
    4: "Winter barley", 5: "Winter rapeseed", 6: "Spring barley",
    7: "Sunflower", 8: "Grapevine", 9: "Beet", 10: "Winter triticale",
    11: "Winter durum wheat", 12: "Fruits, vegetables, flowers",
    13: "Potatoes", 14: "Leguminous fodder", 15: "Soybeans",
    16: "Orchard", 17: "Mixed cereal", 18: "Sorghum", 19: "Void label",
}


def _active_sar_orbits(config):
    legacy_orbit = getattr(config, "sar_satellite", None)
    selection = legacy_orbit or getattr(config, "sar_orbits", "both")
    if selection == "both":
        return ("S1A", "S1D")
    if selection in {"S1A", "S1D"}:
        return (selection,)
    raise ValueError("SAR orbit selection must be 'both', 'S1A', or 'S1D'")


def _mm_satellites(config):
    return [getattr(config, "optical_satellite", "S2"), *_active_sar_orbits(config)]


def _set_mm_experiment_metadata(config):
    if config.model != "mm_ucpmamba":
        return
    modalities = _mm_satellites(config)
    all_channels = {"S2": 10, "S1A": 3, "S1D": 3}
    config.active_modalities = modalities
    config.input_channels = {sat: all_channels[sat] for sat in modalities}
    config.mamba_variant = "Mamba1-SS2D-v01_noz-d_state16-ratio1.0"
    config.experiment_name = "mm_ucpmamba_" + "_".join(s.lower() for s in modalities)
    if os.path.normpath(config.res_dir) == os.path.normpath("./results"):
        config.res_dir = os.path.join(config.res_dir, config.experiment_name)


def _forward_model(model, x, dates, config):
    if config.model != "mm_ucpmamba":
        return model(x, batch_positions=dates)
    if not isinstance(x, dict) or not isinstance(dates, dict):
        raise TypeError("mm_ucpmamba expects a multimodal dict batch")
    optical_satellite = getattr(config, "optical_satellite", "S2")
    sar_orbits = _active_sar_orbits(config)
    if optical_satellite not in x or optical_satellite not in dates:
        raise KeyError("optical satellite is missing from the multimodal batch")
    for orbit in sar_orbits:
        if orbit not in x or orbit not in dates:
            raise KeyError(f"{orbit} is missing from the multimodal batch")
    return model(
        x[optical_satellite],
        x.get("S1A"),
        x.get("S1D"),
        batch_positions_s2=dates.get(optical_satellite),
        batch_positions_s1a=dates.get("S1A"),
        batch_positions_s1d=dates.get("S1D"),
    )


def _diagnostic_ignore_labels(config):
    labels = set()
    if config.ignore_index is not None:
        labels.add(config.ignore_index)
        if config.ignore_index < 0:
            labels.add(config.num_classes + config.ignore_index)
    return {label for label in labels if 0 <= label < config.num_classes}


def _classifier_parameter(model, num_classes):
    candidates = []
    for name, parameter in model.named_parameters():
        if name.endswith("weight") and parameter.ndim == 4 and parameter.shape[0] == num_classes:
            candidates.append((name, parameter))
    return candidates[-1] if candidates else (None, None)


def _classifier_batchnorm(model, num_classes):
    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d) and module.num_features == num_classes:
            candidates.append((name, module))
    return candidates[-1] if candidates else (None, None)


def _diagnostic_update(stats, out, y, config, raw_logits=None):
    class_idx = config.diagnostic_class
    valid = torch.ones_like(y, dtype=torch.bool)
    for label in _diagnostic_ignore_labels(config):
        valid &= y != label
    pred = out.argmax(dim=1)
    gt_class = (y == class_idx) & valid
    pred_class = (pred == class_idx) & valid
    stats["tp"] += int((gt_class & pred_class).sum().item())
    stats["fp"] += int(((~gt_class) & pred_class).sum().item())
    stats["fn"] += int((gt_class & (~pred_class)).sum().item())
    stats["gt_count"] += int(gt_class.sum().item())
    stats["pred_count"] += int(pred_class.sum().item())
    with torch.no_grad():
        # The current ConvBlock applies GroupNorm+ReLU after the final class
        # convolution. Use the forward-hook tensor for true pre-activation logits.
        logits = raw_logits if raw_logits is not None else out
        valid_logits = logits[:, class_idx][valid]
        valid_prob = torch.softmax(logits, dim=1)[:, class_idx][valid]
        if valid_logits.numel():
            other = logits.clone()
            other[:, class_idx] = -float("inf")
            margin = (logits[:, class_idx] - other.max(dim=1).values)[valid]
            stats["logit_max"] = max(stats["logit_max"], float(valid_logits.max().item()))
            stats["logit_sum"] += float(valid_logits.sum().item())
            stats["prob_max"] = max(stats["prob_max"], float(valid_prob.max().item()))
            stats["prob_sum"] += float(valid_prob.sum().item())
            stats["margin_sum"] += float(margin.sum().item())
            stats["valid_count"] += int(valid_logits.numel())
        gt_prob = torch.softmax(logits, dim=1)[:, class_idx][gt_class]
        if gt_prob.numel():
            stats["gt_prob_sum"] += float(gt_prob.sum().item())
            stats["gt_prob_count"] += int(gt_prob.numel())
        effective_prob = torch.softmax(out, dim=1)[:, class_idx][valid]
        effective_other = out.clone()
        effective_other[:, class_idx] = -float("inf")
        effective_margin = (out[:, class_idx] - effective_other.max(dim=1).values)[valid]
        if effective_prob.numel():
            stats["effective_score_max"] = max(
                stats["effective_score_max"], float(out[:, class_idx][valid].max().item())
            )
            stats["effective_score_sum"] += float(out[:, class_idx][valid].sum().item())
            stats["effective_prob_max"] = max(
                stats["effective_prob_max"], float(effective_prob.max().item())
            )
            stats["effective_prob_sum"] += float(effective_prob.sum().item())
            stats["effective_margin_sum"] += float(effective_margin.sum().item())
            stats["effective_valid_count"] += int(effective_prob.numel())
        effective_gt_prob = torch.softmax(out, dim=1)[:, class_idx][gt_class]
        if effective_gt_prob.numel():
            stats["effective_gt_prob_sum"] += float(effective_gt_prob.sum().item())
            stats["effective_gt_prob_count"] += int(effective_gt_prob.numel())


def _diagnostic_metrics(stats, mode):
    union = stats["tp"] + stats["fp"] + stats["fn"]
    valid_count = stats["valid_count"]
    gt_count = stats["gt_prob_count"]
    effective_valid_count = stats["effective_valid_count"]
    effective_gt_count = stats["effective_gt_prob_count"]
    return {
        f"{mode}_potato_iou": 100.0 * stats["tp"] / union if union else 0.0,
        f"{mode}_potato_gt_count": stats["gt_count"],
        f"{mode}_potato_pred_count": stats["pred_count"],
        f"{mode}_potato_tp": stats["tp"],
        f"{mode}_potato_fp": stats["fp"],
        f"{mode}_potato_fn": stats["fn"],
        f"{mode}_potato_logit_max": stats["logit_max"] if valid_count else None,
        f"{mode}_potato_logit_mean": stats["logit_sum"] / valid_count if valid_count else None,
        f"{mode}_potato_prob_max": stats["prob_max"] if valid_count else None,
        f"{mode}_potato_prob_mean": stats["prob_sum"] / valid_count if valid_count else None,
        f"{mode}_potato_prob_mean_gt": stats["gt_prob_sum"] / gt_count if gt_count else None,
        f"{mode}_potato_margin_mean": stats["margin_sum"] / valid_count if valid_count else None,
        f"{mode}_potato_effective_score_max": stats["effective_score_max"] if effective_valid_count else None,
        f"{mode}_potato_effective_score_mean": stats["effective_score_sum"] / effective_valid_count if effective_valid_count else None,
        f"{mode}_potato_effective_prob_max": stats["effective_prob_max"] if effective_valid_count else None,
        f"{mode}_potato_effective_prob_mean": stats["effective_prob_sum"] / effective_valid_count if effective_valid_count else None,
        f"{mode}_potato_effective_prob_mean_gt": stats["effective_gt_prob_sum"] / effective_gt_count if effective_gt_count else None,
        f"{mode}_potato_effective_margin_mean": stats["effective_margin_sum"] / effective_valid_count if effective_valid_count else None,
    }


def iterate(
    model, data_loader, criterion, config, optimizer=None, mode="train", device=None
):
    loss_meter = tnt.meter.AverageValueMeter()
    # 跟踪所有批次的平均损失
    iou_meter = IoU(
        num_classes=config.num_classes,
        ignore_index=config.ignore_index,
        cm_device=config.device,
    )
    diagnostics = {
        "tp": 0, "fp": 0, "fn": 0, "gt_count": 0, "pred_count": 0,
        "logit_max": -float("inf"), "logit_sum": 0.0, "prob_max": -float("inf"),
        "prob_sum": 0.0, "margin_sum": 0.0, "valid_count": 0,
        "gt_prob_sum": 0.0, "gt_prob_count": 0,
        "effective_score_max": -float("inf"), "effective_score_sum": 0.0,
        "effective_prob_max": -float("inf"), "effective_prob_sum": 0.0,
        "effective_margin_sum": 0.0, "effective_valid_count": 0,
        "effective_gt_prob_sum": 0.0, "effective_gt_prob_count": 0,
    }
    classifier_name, classifier_parameter = _classifier_parameter(model, config.num_classes)
    classifier_bn_name, classifier_bn = _classifier_batchnorm(model, config.num_classes)
    classifier_module = None
    raw_logits_holder = {}
    hook_handle = None
    if config.diagnostics and classifier_name is not None:
        classifier_module_name = classifier_name.rsplit(".", 1)[0]
        classifier_module = dict(model.named_modules()).get(classifier_module_name)
        if classifier_module is not None:
            hook_handle = classifier_module.register_forward_hook(
                lambda _module, _inputs, output: raw_logits_holder.__setitem__(
                    "logits", output.detach()
                )
            )
    gradient_sum = 0.0
    classifier_gradient_sum = 0.0
    classifier_potato_gradient_sum = 0.0
    gradient_batches = 0
    nonfinite_gradient_batches = 0
    nonfinite_classifier_gradient_batches = 0
    # 计算多分类任务的交并比和准确率

    t_start = time.time()
    accumulation_steps = getattr(config, "accumulation_steps", 1)
    if mode == "train":
        if optimizer is None:
            raise ValueError("Training requires an optimizer")
        optimizer.zero_grad()
    for i, batch in enumerate(data_loader):
        if device is not None:
            batch = recursive_todevice(batch, device)
        (x, dates), y = batch
        y = y.long()
        # 使用recursive_todevice将数据加载到指定的设备上
        # 将批次数据解包为输入数据x、时间信息dates和标签y
        # 将标签y转为long类型，便于与模型和损失函数兼容

        if mode != "train":
            with torch.no_grad():
                out = _forward_model(model, x, dates, config)
        else:
            out = _forward_model(model, x, dates, config)

        loss = criterion(out, y)
        if mode == "train":
            window_size = accumulation_window_size(
                i, len(data_loader), accumulation_steps
            )
            (loss / window_size).backward()
            update_boundary = is_accumulation_boundary(
                i, len(data_loader), accumulation_steps
            )
            if config.diagnostics and update_boundary:
                total_grad_sq = 0.0
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        total_grad_sq += float(parameter.grad.detach().norm(2).item() ** 2)
                total_grad_norm = total_grad_sq ** 0.5
                if not np.isfinite(total_grad_norm):
                    nonfinite_gradient_batches += 1
                gradient_sum += total_grad_norm
                if classifier_parameter is not None and classifier_parameter.grad is not None:
                    classifier_grad_norm = float(classifier_parameter.grad.detach().norm(2).item())
                    classifier_potato_grad_norm = float(
                        classifier_parameter.grad.detach()[config.diagnostic_class].norm(2).item()
                    )
                    if not np.isfinite(classifier_grad_norm) or not np.isfinite(classifier_potato_grad_norm):
                        nonfinite_classifier_gradient_batches += 1
                    classifier_gradient_sum += classifier_grad_norm
                    classifier_potato_gradient_sum += classifier_potato_grad_norm
                gradient_batches += 1
            if update_boundary:
                optimizer.step()
                optimizer.zero_grad()

        with torch.no_grad():
            pred = out.argmax(dim=1)
            if config.diagnostics:
                _diagnostic_update(
                    diagnostics,
                    out.detach(),
                    y,
                    config,
                    raw_logits=raw_logits_holder.get("logits"),
                )
        iou_meter.add(pred, y)
        # 更新IOU和准确率
        loss_meter.add(loss.item())
        # 跟踪当前批次的损失

        if (i + 1) % config.display_step == 0:
            miou, acc = iou_meter.get_miou_acc()
            print(
                "Step [{}/{}], Loss: {:.4f}, Acc : {:.2f}, mIoU {:.2f}".format(
                    i + 1, len(data_loader), loss_meter.value()[0], acc, miou
                )
            )
    #         每隔display_step 批次打印当前的损失、准确率和平均 IoU

    t_end = time.time()
    total_time = t_end - t_start
    print("Epoch time : {:.1f}s".format(total_time))
    miou, acc = iou_meter.get_miou_acc()
    # 计算最终的准确率和iou
    metrics = {
        "{}_accuracy".format(mode): acc,
        "{}_loss".format(mode): loss_meter.value()[0],
        "{}_IoU".format(mode): miou,
        "{}_epoch_time".format(mode): total_time,
    }
    if config.diagnostics:
        metrics.update(_diagnostic_metrics(diagnostics, mode))
        metrics[f"{mode}_learning_rate"] = (
            float(optimizer.param_groups[0]["lr"]) if optimizer is not None else None
        )
        metrics[f"{mode}_grad_norm_mean"] = gradient_sum / gradient_batches if gradient_batches else None
        metrics[f"{mode}_classifier_name"] = classifier_name
        metrics[f"{mode}_classifier_potato_weight_norm"] = (
            float(classifier_parameter.detach()[config.diagnostic_class].norm(2).item())
            if classifier_parameter is not None else None
        )
        classifier_bias = dict(model.named_parameters()).get(
            classifier_name.rsplit(".", 1)[0] + ".bias" if classifier_name else ""
        )
        metrics[f"{mode}_classifier_potato_bias"] = (
            float(classifier_bias.detach()[config.diagnostic_class].item())
            if classifier_bias is not None else None
        )
        metrics[f"{mode}_classifier_bn_name"] = classifier_bn_name
        metrics[f"{mode}_classifier_bn_potato_gamma"] = (
            float(classifier_bn.weight.detach()[config.diagnostic_class].item())
            if classifier_bn is not None else None
        )
        metrics[f"{mode}_classifier_bn_potato_beta"] = (
            float(classifier_bn.bias.detach()[config.diagnostic_class].item())
            if classifier_bn is not None else None
        )
        metrics[f"{mode}_classifier_grad_norm_mean"] = (
            classifier_gradient_sum / gradient_batches if gradient_batches else None
        )
        metrics[f"{mode}_classifier_potato_grad_norm_mean"] = (
            classifier_potato_gradient_sum / gradient_batches if gradient_batches else None
        )
        metrics[f"{mode}_nonfinite_gradient_batches"] = nonfinite_gradient_batches
        metrics[f"{mode}_nonfinite_classifier_gradient_batches"] = nonfinite_classifier_gradient_batches
    if hook_handle is not None:
        hook_handle.remove()
    # 将当前模式对应的指标汇总为一个字典

    if mode == "test":
        return metrics, iou_meter.conf_metric.value()  # confusion matrix
    else:
        return metrics
#     测试模式返回评估指标和混淆矩阵
# 训练或验证模式：仅返回评估指标

# 用于将数据加载到指定设备上
def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    else:
        return [recursive_todevice(c, device) for c in x]


def dataset_label_counts(dataset, num_classes):
    counts = torch.zeros(num_classes, dtype=torch.long)
    for sample in dataset:
        labels = sample[1].long().reshape(-1)
        valid = (labels >= 0) & (labels < num_classes)
        counts += torch.bincount(labels[valid], minlength=num_classes)
    return counts.tolist()


def save_label_counts(test_fold, train_dataset, val_dataset, config):
    if not config.diagnostics:
        return
    counts = {
        "label_names": PASTIS_LABEL_NAMES,
        "diagnostic_class": config.diagnostic_class,
        "diagnostic_class_name": PASTIS_LABEL_NAMES.get(config.diagnostic_class, "unknown"),
        "num_classes": config.num_classes,
        "train": dataset_label_counts(train_dataset, config.num_classes),
        "val": dataset_label_counts(val_dataset, config.num_classes),
    }
    with open(
        os.path.join(config.res_dir, "Fold_{}".format(test_fold), "diagnostic_label_counts.json"),
        "w",
    ) as outfile:
        json.dump(counts, outfile, indent=4)

# 根据配置参数 config 创建一个结果存储的目录结构，包括主结果目录和用于存储不同折叠（fold）的子目录。
def prepare_output(config):
    os.makedirs(config.res_dir, exist_ok=True)
    for fold in range(1, 6):
        os.makedirs(os.path.join(config.res_dir, "Fold_{}".format(fold)), exist_ok=True)

# 将日志信息保存为 JSON 文件，用于记录训练或验证过程中的关键数据，便于后续分析或断点恢复。
def checkpoint(fold, log, config):
    with open(
        os.path.join(config.res_dir, "Fold_{}".format(fold), "trainlog.json"), "w"
    ) as outfile:
        json.dump(log, outfile, indent=4)


def save_results(fold, metrics, conf_mat, config):
    with open(
        os.path.join(config.res_dir, "Fold_{}".format(fold), "test_metrics.json"), "w"
    ) as outfile:
        json.dump(metrics, outfile, indent=4)
    pkl.dump(
        conf_mat,
        open(
            os.path.join(config.res_dir, "Fold_{}".format(fold), "conf_mat.pkl"), "wb"
        ),
    )

# overall_performance 函数的作用是计算并保存所有折叠（fold）的整体性能指标。具体步骤如下：
# 加载并累加混淆矩阵：
# 从各折叠目录中读取混淆矩阵文件，并将它们相加，得到整个数据集的总混淆矩阵。
# 删除忽略的类别（可选）：
# 如果配置中指定了 ignore_index，则从混淆矩阵中移除对应的行和列。
# 分析混淆矩阵：
# 调用 confusion_matrix_analysis，计算整体性能指标（如准确率、IoU 等）。
# 打印整体性能：
# 输出整体准确率（Accuracy）和宏平均 IoU（MACRO_IoU）。
# 保存性能指标到文件：
# 将性能指标写入到 overall.json 文件，便于后续查看或处理。
def overall_performance(config):
    cm = np.zeros((config.num_classes, config.num_classes))
    for fold in range(1, 6):
        cm += pkl.load(
            open(
                os.path.join(config.res_dir, "Fold_{}".format(fold), "conf_mat.pkl"),
                "rb",
            )
        )

    if config.ignore_index is not None:
        cm = np.delete(cm, config.ignore_index, axis=0)
        cm = np.delete(cm, config.ignore_index, axis=1)

    _, perf = confusion_matrix_analysis(cm)

    print("Overall performance:")
    print("Acc: {},  IoU: {}".format(perf["Accuracy"], perf["MACRO_IoU"]))

    with open(os.path.join(config.res_dir, "overall.json"), "w") as file:
        file.write(json.dumps(perf, indent=4))


def main(config):
    # 定义了一个名为 fold_sequence 的列表，表示数据集的折叠分割顺序。这通常用于交叉验证（Cross-Validation）的实现。
    # train_folds：用于训练的数据折叠序列。
    # val_folds：用于验证的数据折叠序列。
    # test_folds：用于测试的数据折叠序列。

    np.random.seed(config.rdm_seed)
    torch.manual_seed(config.rdm_seed)
    config.accumulation_steps = getattr(config, "accumulation_steps", 1)
    if config.accumulation_steps < 1:
        raise ValueError("accumulation_steps must be at least 1")
    config.effective_batch_size = config.batch_size * config.accumulation_steps
    _set_mm_experiment_metadata(config)
    prepare_output(config)
    print(
        "Micro-batch {}, accumulation {}, effective batch {}".format(
            config.batch_size,
            config.accumulation_steps,
            config.effective_batch_size,
        )
    )
    # 调用 prepare_output 函数，创建存储结果的目录结构
    device = torch.device(config.device)

    test_folds = sorted(FOLD_SPLITS) if config.fold is None else [config.fold]
    # 如果指定单独的折如config.fold=3时，fold_sequence变为[[[3, 4, 5], [1], [2]]]
    for test_fold in test_folds:
        train_folds, val_fold, _ = get_fold_split(test_fold)
        #  遍历 fold_sequence：train_folds：当前折的训练数据集。
        # val_fold：验证数据集。test_fold：测试数据集。
        # 如果指定了折，fold 被固定为对应索引值（config.fold - 1）。

        # Dataset definition
        sats = _mm_satellites(config) if config.model == "mm_ucpmamba" else ["S2"]
        dt_args = dict(
            folder=config.dataset_folder,
            norm=True,
            reference_date=config.ref_date,
            mono_date=config.mono_date,
            target="semantic",
            sats=sats,
        )
        # 单日期模式，设置为 None 表示多时序数据。

        # **dt_args: 传入通用的参数，包括数据集路径、归一化设置、基准日期、模式等。
        # 默认情况下不做时间 dropout（设为 None），也就是数据集会使用完整的时间序列。
        temporal_dropout = None
        # 只有当选用的模型是多模态 UCPMamba（mm_ucpmamba）时，才启用时间 dropout，并且为三种模态分别从命令行/配置里读取各自的丢弃概率，组成一个字典。
        if config.model == "mm_ucpmamba":
            temporal_dropout = {
                "S2": config.temporal_dropout_s2,
                "S1A": config.temporal_dropout_s1a,
                "S1D": config.temporal_dropout_s1d,
            }
        #最后用这些参数构造训练用的 PASTIS 数据集对象。
        dt_train = PASTIS_Dataset(
            **dt_args,
            folds=train_folds,
            cache=config.cache,
            temporal_dropout=temporal_dropout,
        )
        dt_val = PASTIS_Dataset(**dt_args, folds=val_fold, cache=config.cache)
        dt_test = PASTIS_Dataset(**dt_args, folds=[test_fold])
        torch.cuda.empty_cache()

        collate_fn = lambda x: utils.pad_collate(x, pad_value=config.pad_value)
        # 使用自定义的 pad_collate 函数对批数据进行动态填充（padding）。
        # pad_value：填充值，避免时间步长不一致导致的错误。
        train_loader = data.DataLoader(
            dt_train,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
        # dataset: 数据集实例。
        # batch_size: 批大小，使用配置中的值。
        # shuffle: 是否打乱数据。
        # drop_last: 是否丢弃最后一个不足批大小的数据。
        # collate_fn: 使用自定义的填充函数。
        val_loader = data.DataLoader(
            dt_val,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )
        test_loader = data.DataLoader(
            dt_test,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        if config.diagnostics:
            class_name = PASTIS_LABEL_NAMES.get(config.diagnostic_class, "unknown")
            print(
                "Diagnostics: class {} = {}, num_classes={}, ignore_labels={}".format(
                    config.diagnostic_class,
                    class_name,
                    config.num_classes,
                    sorted(_diagnostic_ignore_labels(config)),
                )
            )
            save_label_counts(test_fold, dt_train, dt_val, config)

        print(
            "Train {}, Val {}, Test {}".format(len(dt_train), len(dt_val), len(dt_test))
        )

        # Model definition
        model = model_utils.get_model(config, mode="semantic")
        # config：包含模型相关的超参数（如网络架构、层宽度等）。
        # mode="semantic"：设置模型的任务模式为语义分割。
        config.N_params = utils.get_ntrainparams(model)
        # 计算模型中可训练参数的总数。
        with open(os.path.join(config.res_dir, "conf.json"), "w") as file:
            file.write(json.dumps(vars(config), indent=4))
        #     将配置文件转化为 JSON 格式，便于记录和分析。
        # 使用 vars(config) 提取 config 对象的字典表示
        print(model)
        print("TOTAL TRAINABLE PARAMETERS :", config.N_params)
        print("Trainable layers:")
        for name, p in model.named_parameters():
            if p.requires_grad:
                print(name)
        #         确保所有期望训练的层都被正确设置为 requires_grad=True。
        # 用于调试冻结或解冻特定层的问题
        model = model.to(device)
        model.apply(weight_init)
        if config.model == "mm_ucpmamba":
            model.reset_quality_fusion_initialization()

        # Optimizer and Loss
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        weights = torch.ones(config.num_classes, device=device).float()
        weights[config.ignore_index] = 0
        criterion = nn.CrossEntropyLoss(weight=weights)

        # Training loop
        trainlog = {}
        # 用于记录每个 epoch 的训练和验证指标。
        best_mIoU = 0
        # best_mIoU：存储最佳验证集上的平均交并比 (mIoU)，用于保存最佳模型。
        for epoch in range(1, config.epochs + 1):
            torch.cuda.empty_cache()  # 清理未使用的缓存内存
            print("EPOCH {}/{}".format(epoch, config.epochs))

            model.train()
            train_metrics = iterate(
                model,
                data_loader=train_loader,
                criterion=criterion,
                config=config,
                optimizer=optimizer,
                mode="train",
                device=device,
            )
            if epoch % config.val_every == 0 and epoch > config.val_after:
                # 在设定的间隔（config.val_every）进行验证。
                print("Validation . . . ")
                model.eval()
                val_metrics = iterate(
                    model,
                    data_loader=val_loader,
                    criterion=criterion,
                    config=config,
                    optimizer=optimizer,
                    mode="val",
                    device=device,
                )

                print(
                    "Loss {:.4f},  Acc {:.2f},  IoU {:.4f}".format(
                        val_metrics["val_loss"],
                        val_metrics["val_accuracy"],
                        val_metrics["val_IoU"],
                    )
                )

                trainlog[epoch] = {**train_metrics, **val_metrics}
                checkpoint(test_fold, trainlog, config)
                if val_metrics["val_IoU"] >= best_mIoU:
                    best_mIoU = val_metrics["val_IoU"]
                    torch.save(
                        {
                            "epoch": epoch,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "model_metadata": {
                                "model_name": config.model,
                                "active_modalities": getattr(config, "active_modalities", ["S2"]),
                                "input_channels": getattr(config, "input_channels", {"S2": 10}),
                                "fusion_mode": getattr(config, "fusion_mode", None),
                                "mamba_variant": getattr(config, "mamba_variant", None),
                            },
                        },
                        os.path.join(
                            config.res_dir, "Fold_{}".format(test_fold), "model.pth.tar"
                        ),
                    )
            else:
                trainlog[epoch] = {**train_metrics}
                checkpoint(test_fold, trainlog, config)
        #         通过 checkpoint 函数保存训练日志到文件中。
        # 记录每个 epoch 的训练和验证结果，便于后续分析

        print("Testing best epoch . . .")
        model.load_state_dict(
            torch.load(
                os.path.join(
                    config.res_dir, "Fold_{}".format(test_fold), "model.pth.tar"
                )
            )["state_dict"],
            strict=True,
        )
        model.eval()

        test_metrics, conf_mat = iterate(
            model,
            data_loader=test_loader,
            criterion=criterion,
            config=config,
            optimizer=optimizer,
            mode="test",
            device=device,
        )
        print(
            "Loss {:.4f},  Acc {:.2f},  IoU {:.4f}".format(
                test_metrics["test_loss"],
                test_metrics["test_accuracy"],
                test_metrics["test_IoU"],
            )
        )
        save_results(test_fold, test_metrics, conf_mat.cpu().numpy(), config)

    if config.fold is None:
        overall_performance(config)


if __name__ == "__main__":
    config = parser.parse_args()
    for k, v in vars(config).items():
        if k in list_args and v is not None:
            v = v.replace("[", "")
            v = v.replace("]", "")
            config.__setattr__(k, list(map(int, v.split(","))))
    #         将 config 对象转换为字典，方便逐项处理。
    # 参数名 k 是否属于需要特殊处理的列表 list_args。
    # 参数值 v 是否非空。去除方括号（[ 和 ]）。
    # 按逗号分隔字符串，并将每个部分转换为整数。
    # 使用 config.__setattr__ 更新参数值。

    assert config.num_classes == config.out_conv[-1]
    if config.diagnostics:
        if not 0 <= config.diagnostic_class < config.num_classes:
            raise ValueError("diagnostic_class must be within [0, num_classes).")
        if config.diagnostic_class != 13:
            print("Warning: diagnostic_class {} is not the standard PASTIS Potatoes index 13.".format(config.diagnostic_class))
    # 确保输出层的通道数与目标类别数一致

    pprint.pprint(config)
    main(config)
