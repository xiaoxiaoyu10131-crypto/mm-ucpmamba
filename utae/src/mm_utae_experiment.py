"""MM-UTAE 的配置校验、模型构造与论文辅助监督。"""

import math


def validate_mm_utae_config(config):
    """拒绝会被 MM-UTAE 静默忽略的其他模型设置。"""
    if getattr(config, "fusion_mode", "concat") != "concat":
        raise ValueError("mm_utae implements late concat fusion only")
    if getattr(config, "optical_satellite", "S2") != "S2":
        raise ValueError("mm_utae requires the S2 optical modality")
    if getattr(config, "use_quality_temporal", False):
        raise ValueError("mm_utae does not use quality-guided temporal attention")
    if getattr(config, "modality_dropout", 0.0) != 0:
        raise ValueError("mm_utae uses temporal dropout, not modality dropout")
    if getattr(config, "decoder_norm", "batch") != "batch":
        raise ValueError("mm_utae retains the official U-TAE BatchNorm decoder")
    weight = getattr(config, "mm_utae_aux_weight", 0.0)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("mm_utae_aux_weight must be finite and non-negative")
    head = getattr(config, "mm_utae_head", "legacy")
    if head not in {"legacy", "logits"}:
        raise ValueError("mm_utae_head must be legacy or logits")
    if head == "logits" and getattr(config, "final_bn_init", "legacy") == "ones":
        raise ValueError("raw logits have no final BatchNorm; use --final-bn-init legacy")
    if len(config.out_conv) != 2 or config.out_conv[-1] != config.num_classes:
        raise ValueError("mm_utae needs two out_conv widths ending in num_classes")


def build_mm_utae(config):
    """仅创建 MM-UTAE；不导入任何 Mamba/CUDA 扩展。"""
    from src.backbones.mm_utae import MMUTAE

    validate_mm_utae_config(config)
    return MMUTAE(
        optical_input_dim=getattr(config, "optical_input_dim", 10),
        sar_input_dim=getattr(config, "sar_input_dim", 3),
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
        pad_value=config.pad_value,
        padding_mode=config.padding_mode,
        sar_orbits=getattr(config, "sar_satellite", None)
        or getattr(config, "sar_orbits", "both"),
        auxiliary=getattr(config, "mm_utae_aux_weight", 0.0) > 0,
        head_mode=getattr(config, "mm_utae_head", "legacy"),
    )


def mm_utae_loss(result, target, criterion, auxiliary_weight=0.0):
    """L = CE(fusion) + lambda * sum_m CE(modality_m)。"""
    if not isinstance(result, dict):
        if auxiliary_weight > 0:
            raise ValueError("auxiliary supervision requires per-modality logits")
        return result, criterion(result, target)
    logits = result["logits"]
    loss = criterion(logits, target)
    if auxiliary_weight > 0:
        auxiliary = [value for key, value in result.items() if key.startswith("logits_")]
        if not auxiliary:
            raise ValueError("auxiliary supervision requires per-modality logits")
        loss = loss + auxiliary_weight * sum(criterion(value, target) for value in auxiliary)
    return logits, loss


def initialize_mm_utae_heads(model, num_classes):
    """逐个初始化融合/辅助分类头，避免误选最后一个辅助头。"""
    from src.learning.weight_init import reset_final_batchnorm_identity

    names = []
    heads = {"fusion_head": model.fusion_head, **dict(model.aux_heads.items())}
    for name, head in heads.items():
        names.append(f"{name}.{reset_final_batchnorm_identity(head, num_classes)}")
    return names
