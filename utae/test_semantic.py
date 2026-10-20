"""
Script for semantic inference with pre-trained models
Author: Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""
import argparse
from functools import partial
import json
import os
import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from src import utils, model_utils
from src.dataset import PASTIS_Dataset
from src.mm_collate import mid_pad_collate
from src.mm_async_collate import async_pad_collate
from src.mm_async_experiment import add_async_arguments, check_async_checkpoint
from src.folds import FOLD_SPLITS, get_fold_split

from train_semantic import (
    _is_mm_model,
    _mm_satellites,
    _set_mm_experiment_metadata,
    iterate,
    overall_performance,
    save_results,
    prepare_output,
)

parser = argparse.ArgumentParser()
# Model parameters
parser.add_argument(
    "--model",
    default=None,
    choices=[
        "utae", "unet3d", "fpn", "convlstm", "convgru", "uconvlstm",
        "ucpmamba", "ucpmamba_vmambav2", "ucpmamba_mamba2",
        "ucpmamba_directvss",
        "ucpmamba_mean", "ucpmamba_plainlstm", "mm_ucpmamba", "mm_ucpmamba_mid", "mm_ucpmamba_async", "mm_utae", "buconvlstm",
    ],
    help="Override the model name stored in conf.json",
)
parser.add_argument(
    "--weight_folder",
    type=str,
    default="",
    help="Path to the main folder containing the pre-trained weights",
)
parser.add_argument(
    "--dataset_folder",
    default="",
    type=str,
    help="Path to the folder where the results are saved.",
)
parser.add_argument(
    "--res_dir",
    default="./inference_utae",
    type=str,
    help="Path to directory where results are written."
)
parser.add_argument(
    "--num_workers", default=8, type=int, help="Number of data loading workers"
)
parser.add_argument(
    "--fold",
    default=None,
    type=int,
    choices=range(1, 6),
    help="Official PASTIS test Fold (between 1 and 5)",
)
parser.add_argument(
    "--checkpoint_fold",
    default=None,
    type=int,
    help="Legacy checkpoint directory index; defaults to the official test Fold",
)
parser.add_argument(
    "--device",
    default="cuda",
    type=str,
    help="Name of device to use for tensor computations (cuda/cpu)",
)
parser.add_argument(
    "--display_step",
    default=50,
    type=int,
    help="Interval in batches between display of training metrics",
)
parser.add_argument(
    "--sar-orbits",
    dest="sar_orbits",
    choices=["both", "S1A", "S1D"],
    default=None,
    help="Override the SAR orbit selection stored in conf.json.",
)
parser.add_argument(
    "--sar-satellite",
    dest="sar_satellite",
    choices=["S1A", "S1D"],
    default=None,
    help="Backward-compatible single-orbit override.",
)
add_async_arguments(parser, inference=True)
parser.add_argument("--mid-num-bins", default=None, type=int)
parser.add_argument("--mid-low-memory", action="store_true", default=None)
parser.add_argument("--mid-start-day", default=None, type=float)
parser.add_argument("--mid-end-day", default=None, type=float)

def main(config):
    np.random.seed(config.rdm_seed)
    torch.manual_seed(config.rdm_seed)
    _set_mm_experiment_metadata(config)
    device = torch.device(config.device)
    prepare_output(config)
    with open(os.path.join(config.res_dir, "conf.json"), "w") as file:
        json.dump(vars(config), file, indent=4)

    model = model_utils.get_model(config, mode="semantic")
    model = model.to(device)

    config.N_params = utils.get_ntrainparams(model)
    print(model)
    print("TOTAL TRAINABLE PARAMETERS :", config.N_params)

    test_folds = sorted(FOLD_SPLITS) if config.fold is None else [config.fold]
    for requested_test_fold in test_folds:
        _, _, test_fold_ids = get_fold_split(requested_test_fold)
        test_fold = test_fold_ids[0]
        checkpoint_fold = getattr(config, "checkpoint_fold", None) or test_fold

        # Dataset definition
        sats = _mm_satellites(config) if _is_mm_model(config) else ["S2"]
        dt_test = PASTIS_Dataset(
            folder=config.dataset_folder,
            norm=True,
            reference_date=config.ref_date,
            mono_date=config.mono_date,
            target="semantic",
            sats=sats,
            folds=test_fold_ids,
        )
        collate_fn = lambda x: utils.pad_collate(x, pad_value=config.pad_value)
        if config.model in {"mm_ucpmamba_mid", "mm_ucpmamba_async"}:
            collate_fn = partial(mid_pad_collate, pad_value=config.pad_value)
        if config.model == "mm_ucpmamba_async":
            collate_fn = partial(async_pad_collate, pad_value=config.pad_value)
        test_loader = data.DataLoader(
            dt_test,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        # Load weights
        sd = torch.load(
            os.path.join(
                config.weight_folder,
                "Fold_{}".format(checkpoint_fold),
                "model.pth.tar",
            ),
            map_location=device,
        )
        check_async_checkpoint(sd, config)
        model.load_state_dict(sd["state_dict"], strict=True)

        # Loss
        weights = torch.ones(config.num_classes, device=device).float()
        weights[config.ignore_index] = 0
        criterion = nn.CrossEntropyLoss(weight=weights)

        # Inference
        print("Testing . . .")
        model.eval()
        test_metrics, conf_mat = iterate(
            model,
            data_loader=test_loader,
            criterion=criterion,
            config=config,
            optimizer=None,
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
        save_results(test_fold, test_metrics, conf_mat, config)

    if config.fold is None:
        overall_performance(config)


if __name__ == "__main__":
    test_config = parser.parse_args()


    with open(os.path.join(test_config.weight_folder, "conf.json")) as file:
        model_config = json.loads(file.read())

    cli_config = {
        key: value
        for key, value in vars(test_config).items()
        if value is not None and key not in {"model", "fold", "checkpoint_fold"}
    }
    if test_config.model is not None:
        cli_config["model"] = test_config.model
    config = {**model_config, **cli_config}
    config = argparse.Namespace(**config)
    config.fold = test_config.fold
    config.checkpoint_fold = test_config.checkpoint_fold

    pprint.pprint(config)
    main(config)
