from src.backbones import utae


def __getattr__(name):
    """按需保留旧调用方对 backbone 模块属性的访问。"""
    if name in {"unet3d", "convlstm", "convgru", "fpn", "ucpmamba", "uconvlstm", "mm_ucpmamba"}:
        from importlib import import_module

        module = import_module(f"src.backbones.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_model(config, mode="semantic"):
    if mode == "semantic" and config.model == "mm_ucpmamba_async":
        from src.mm_async_experiment import build_async

        return build_async(config)

    if mode == "semantic" and config.model == "mm_utae":
        from src.mm_utae_experiment import build_mm_utae

        return build_mm_utae(config)

    from src.backbones import unet3d, convlstm, convgru, fpn, ucpmamba, uconvlstm, mm_ucpmamba

    if mode == "semantic":
        if config.model == "utae":
            model = utae.UTAE(
                input_dim=10,
                encoder_widths=config.encoder_widths,
                decoder_widths=config.decoder_widths,
                out_conv=config.out_conv,
                str_conv_k=config.str_conv_k,
                str_conv_s=config.str_conv_s,
                str_conv_p=config.str_conv_p,
                agg_mode=config.agg_mode,
                encoder_norm=config.encoder_norm,
                n_head=config.n_head,
                d_model=config.d_model,
                d_k=config.d_k,
                encoder=False,
                return_maps=False,
                pad_value=config.pad_value,
                padding_mode=config.padding_mode,
            )
        elif config.model == "unet3d":
            model = unet3d.UNet3D(
                in_channel=10, n_classes=config.num_classes, pad_value=config.pad_value
            )
        elif config.model == "fpn":
            model = fpn.FPNConvLSTM(
                input_dim=10,
                num_classes=config.num_classes,
                inconv=[32, 64],
                n_levels=4,
                n_channels=64,
                hidden_size=88,
                input_shape=(128, 128),
                mid_conv=True,
                pad_value=config.pad_value,
            )
        elif config.model == "convlstm":
            model = convlstm.ConvLSTM_Seg(
                num_classes=config.num_classes,
                input_size=(128, 128),
                input_dim=10,
                kernel_size=(3, 3),
                hidden_dim=160,
            )
        elif config.model == "convgru":
            model = convgru.ConvGRU_Seg(
                num_classes=config.num_classes,
                input_size=(128, 128),
                input_dim=10,
                kernel_size=(3, 3),
                hidden_dim=180,
            )
        elif config.model == "uconvlstm":
            model = uconvlstm.UConvLSTM(
                input_dim=10,
                encoder_widths=config.encoder_widths,
                decoder_widths=config.decoder_widths,
                out_conv=config.out_conv,
                str_conv_k=config.str_conv_k,
                str_conv_s=config.str_conv_s,
                str_conv_p=config.str_conv_p,
                temporal="lstm",
                input_size=128,
                encoder_norm=config.encoder_norm,
                hidden_dim=64,
                encoder=False,
                padding_mode=config.padding_mode,
                pad_value=config.pad_value,
            )
        elif config.model in {"mm_ucpmamba", "mm_ucpmamba_mid"}:
            sar_orbits = getattr(config, "sar_satellite", None) or getattr(
                config, "sar_orbits", "both"
            )
            model_class = mm_ucpmamba.MMUCPMamba
            extra_kwargs = dict(
                use_complementary_residual=getattr(config, "use_complementary_residual", True),
                use_quality_temporal=getattr(config, "use_quality_temporal", False),
                quality_lambda=getattr(config, "quality_lambda", 1.0),
            )
            if config.model == "mm_ucpmamba_mid":
                from src.backbones.mm_ucpmamba_mid import MMUCPMambaMid

                if getattr(config, "use_quality_temporal", False):
                    raise ValueError("mm_ucpmamba_mid is the quality-free concat baseline")
                model_class = MMUCPMambaMid
                extra_kwargs = dict(
                    num_bins=getattr(config, "mid_num_bins", 24),
                    start_day=getattr(config, "mid_start_day", None),
                    end_day=getattr(config, "mid_end_day", None),
                    low_memory=getattr(config, "mid_low_memory", False),
                )
            model = model_class(
                optical_input_dim=getattr(config, "optical_input_dim", 10),
                sar_input_dim=getattr(config, "sar_input_dim", 3),
                encoder_widths=config.encoder_widths,
                decoder_widths=config.decoder_widths,
                out_conv=config.out_conv,
                str_conv_k=config.str_conv_k,
                str_conv_s=config.str_conv_s,
                str_conv_p=config.str_conv_p,
                input_size=128,
                encoder_norm=config.encoder_norm,
                decoder_norm=getattr(config, "decoder_norm", "batch"),
                decoder_groups=getattr(config, "decoder_groups", 8),
                hidden_dim=64,
                padding_mode=config.padding_mode,
                pad_value=config.pad_value,
                sar_orbits=sar_orbits,
                #"concat"：通道拼接再压回；"qaf"：质量感知加权融合
                fusion_mode=getattr(config, "fusion_mode", "concat"),
                #训练时随机丢弃某一模态特征（不会同时丢两边），增强鲁棒性
                modality_dropout=getattr(config, "modality_dropout", 0.0),
                **extra_kwargs,
            )
        elif config.model in {
            "ucpmamba",
            "ucpmamba_vmambav2",
            "ucpmamba_mamba2",
            "ucpmamba_directvss",
            "ucpmamba_mean",
            "ucpmamba_plainlstm",
        }:
            temporal_encoder_classes = {
                "ucpmamba_plainlstm": uconvlstm.PlainConvLSTM,
                "ucpmamba_vmambav2": convlstm.VMambaV2ConvLSTM,
                "ucpmamba_mamba2": convlstm.Mamba2ConvLSTM,
                "ucpmamba_directvss": convlstm.DirectVSSConvLSTM,
            }
            model = ucpmamba.UCPMamba(
                input_dim=10,
                encoder_widths=config.encoder_widths,
                decoder_widths=config.decoder_widths,
                out_conv=config.out_conv,
                str_conv_k=config.str_conv_k,
                str_conv_s=config.str_conv_s,
                str_conv_p=config.str_conv_p,
                temporal="lstm",
                input_size=128,
                encoder_norm=config.encoder_norm,
                hidden_dim=64,
                encoder=False,
                padding_mode=config.padding_mode,
                pad_value=config.pad_value,
                temporal_encoder_class=temporal_encoder_classes.get(
                    config.model, convlstm.ConvLSTM
                ),
                skip_aggregation=(
                    "phenology"
                    if config.model == "ucpmamba_plainlstm"
                    else (
                        "mean"
                        if config.model == "ucpmamba_mean"
                        else "phenology"
                    )
                ),
            )
        elif config.model == "buconvlstm":
            model = utae.RecUNet(
                input_dim=10,
                encoder_widths=[64, 64, 64, 128],
                decoder_widths=[32, 32, 64, 128],
                out_conv=[32, 20],
                str_conv_k=4,
                str_conv_s=2,
                str_conv_p=1,
                temporal="lstm",
                input_size=128,
                encoder_norm="group",
                hidden_dim=30,
                encoder=False,
                padding_mode="zeros",
                pad_value=0,
            )
        return model
    elif mode == "panoptic":
        from src.panoptic import paps

        if config.backbone == "utae":
            model = utae.UTAE(
                input_dim=10,
                encoder_widths=config.encoder_widths,
                decoder_widths=config.decoder_widths,
                out_conv=config.out_conv,
                str_conv_k=config.str_conv_k,
                str_conv_s=config.str_conv_s,
                str_conv_p=config.str_conv_p,
                agg_mode=config.agg_mode,
                encoder_norm=config.encoder_norm,
                n_head=config.n_head,
                d_model=config.d_model,
                d_k=config.d_k,
                encoder=True,
                return_maps=False,
                pad_value=config.pad_value,
                padding_mode=config.padding_mode,
            )
        elif config.backbone == "uconvlstm":
            model = uconvlstm.UConvLSTM(
                input_dim=10,
                encoder_widths=[64, 64, 64, 128],
                decoder_widths=[32, 32, 64, 128],
                out_conv=[32, 20],
                str_conv_k=4,
                str_conv_s=2,
                str_conv_p=1,
                temporal="lstm",
                input_size=128,
                encoder_norm="group",
                hidden_dim=64,
                encoder=True,
                padding_mode="zeros",
                pad_value=0,
            )
        else:
            raise NotImplementedError

        model = paps.PaPs(
            encoder=model,
            num_classes=config.num_classes,
            shape_size=config.shape_size,
            mask_conv=config.mask_conv,
            min_confidence=config.min_confidence,
            min_remain=config.min_remain,
            mask_threshold=config.mask_threshold,
        )
        return model
    else:
        raise NotImplementedError
