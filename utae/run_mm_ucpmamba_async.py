"""Async 设计的独立训练入口；其余 CLI 和训练循环复用现有实现。"""

import copy
import json
import sys

from train_semantic import list_args, main, parser


def make_config(arguments):
    training_parser = copy.deepcopy(parser)
    training_parser.set_defaults(
        model="mm_ucpmamba_async", encoder_widths="[64,64,64,128]",
        decoder_widths="[32,32,64,128]", out_conv="[32,20]",
        encoder_norm="group", decoder_norm="group", decoder_groups=8,
        final_bn_init="ones", sar_orbits="both", fusion_mode="concat",
        mid_num_bins=24, mid_start_day=-200., mid_end_day=600., mid_low_memory=True,
        ref_date="2018-09-01", batch_size=4, accumulation_steps=1,
        epochs=100, lr=.001, rdm_seed=1, modality_dropout=0.,
        temporal_dropout_s2=.4, temporal_dropout_s1a=.2, temporal_dropout_s1d=.2,
        num_classes=20, ignore_index=19,
    )
    config = training_parser.parse_args(arguments)
    if config.model != "mm_ucpmamba_async":
        raise ValueError("this entry point only runs mm_ucpmamba_async")
    for name in list_args:
        value = getattr(config, name)
        if isinstance(value, str):
            setattr(config, name, json.loads(value))
    return config


if __name__ == "__main__":
    main(make_config(sys.argv[1:]))
