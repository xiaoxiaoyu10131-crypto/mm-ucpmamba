"""108 共享主路上的窗口动态表征与真实时差邻窗交互。"""

import torch
from torch import nn

from src.backbones.mm_async_interaction import LocalAsyncInteraction
from src.backbones.mm_dynamic_window import DynamicWindowRepresentation
from src.backbones.mm_ucpmamba_mid import MMUCPMambaMid


class MMUCPMambaAsync(MMUCPMambaMid):
    def __init__(self, *args, async_radius=1, async_dim=16, trend_min_span=2.0,
                 dynamic_enabled=True, interaction_enabled=True, residual_init=0.0, **kwargs):
        kwargs.setdefault("start_day", -200)
        kwargs.setdefault("end_day", 600)
        kwargs.setdefault("decoder_norm", "group")
        super().__init__(*args, **kwargs)
        if self.active_modalities != ("S2", "S1A", "S1D"):
            raise ValueError("mm_ucpmamba_async requires S2, S1A and S1D")
        if residual_init != 0:
            raise ValueError("async residual coefficients must start at zero")
        self.dynamic_windows = nn.ModuleDict({
            m: DynamicWindowRepresentation(self.encoder_widths[2], 16, trend_min_span, dynamic_enabled)
            for m in self.active_modalities
        })
        self.async_interaction = LocalAsyncInteraction(
            self.active_modalities, self.encoder_widths[2], async_dim, async_radius, interaction_enabled
        )
        self.last_diagnostics = {}
        self._async_settings = (async_radius, async_dim, trend_min_span,
                                dynamic_enabled, interaction_enabled, residual_init)
        self.register_buffer("async_signature", torch.tensor(self._async_settings, dtype=torch.float64))

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self.async_signature = torch.tensor(self._async_settings, dtype=torch.float64,
                                            device=self.async_signature.device)
        return self

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        saved = state_dict.get(prefix + "async_signature")
        if saved is not None and not torch.equal(saved.cpu(), self.async_signature.cpu()):
            error_msgs.append("async configuration does not match checkpoint signature")
            return
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def reset_async_initialization(self):
        """通用初始化之后调用，恢复两个新模块的零残差起点。"""
        for module in self.dynamic_windows.values():
            nn.init.zeros_(module.gamma)
        for gamma in self.async_interaction.gamma.values():
            nn.init.zeros_(gamma)

    def _align_modality(self, name, features, positions, pads, valid):
        return self.dynamic_windows[name](features, positions, pads, valid, self.aligner)

    def _interact_aligned(self, aligned, valid, info):
        for name in self.active_modalities:
            if name not in info:
                shape = valid[name].shape
                info[name] = {
                    "mean_dates": aligned[name].new_zeros(shape, dtype=torch.float64),
                    "info": aligned[name].new_zeros(shape[0], shape[1], 5, *shape[-2:]),
                    "trend_valid": valid[name].clone(),
                }
        width = (self.aligner.end_day - self.aligner.start_day) / self.aligner.num_bins
        result, diagnostic = self.async_interaction(aligned, valid, info, width)
        diagnostic["observed_windows"] = sum(mask.sum() for mask in valid.values()).detach()
        diagnostic["trend_windows"] = sum(item["trend_valid"].sum() for item in info.values()).detach()
        for name in self.active_modalities:
            diagnostic[f"gamma_A_{name}"] = self.dynamic_windows[name].gamma.detach().clone()
            diagnostic[f"gamma_B_{name}"] = self.async_interaction.gamma[name].detach().clone()
        self.last_diagnostics = diagnostic
        return result, diagnostic
