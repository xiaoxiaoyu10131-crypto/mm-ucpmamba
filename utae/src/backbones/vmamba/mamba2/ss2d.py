"""Official-VMamba-style SS2Dm0 with a BCHW boundary adapter."""

import math
from functools import partial

import torch
import torch.nn as nn

from .ssd_minimal import selective_scan_chunk_fn


def _cross_scan(x: torch.Tensor) -> torch.Tensor:
    """BHWC -> four row/column sequences, shaped B,L,4,C."""
    batch, height, width, channels = x.shape
    row = x.reshape(batch, height * width, channels)
    column = x.transpose(1, 2).contiguous().reshape(batch, height * width, channels)
    return torch.stack((row, column, row.flip(1), column.flip(1)), dim=2)


def _cross_merge(
    y: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """Merge four B,L,4,C scan directions back to BHWC."""
    batch, _, _, channels = y.shape
    row = y[:, :, 0]
    column = y[:, :, 1].reshape(batch, width, height, channels)
    column = column.transpose(1, 2).contiguous().reshape(batch, height * width, channels)
    reverse_row = y[:, :, 2].flip(1)
    reverse_column = y[:, :, 3].flip(1).reshape(batch, width, height, channels)
    reverse_column = reverse_column.transpose(1, 2).contiguous().reshape(
        batch, height * width, channels
    )
    return (row + column + reverse_row + reverse_column).reshape(
        batch, height, width, channels
    )


class SS2Dm0(nn.Module):
    """Mamba2/SSD SS2D core following official VMamba's ``SS2Dm0``."""

    def __init__(
        self,
        d_model=96,
        d_state=64,
        ssm_ratio=1.0,
        dt_rank="auto",
        act_layer=nn.GELU,
        d_conv=3,
        conv_bias=False,
        dropout=0.0,
        bias=False,
        initialize="v2",
        forward_type="m0_noz",
        channel_first=True,
        with_initial_state=False,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        tags = forward_type.split("_")
        if tags[0] != "m0":
            raise ValueError(f"SS2Dm0 only supports m0 forward types: {forward_type}")

        self.channel_first = channel_first
        self.disable_z = "noz" in tags
        self.disable_z_act = "nozact" in tags
        self.out_act = nn.GELU() if "oact" in tags else nn.Identity()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)
        if self.d_inner % self.dt_rank != 0:
            raise ValueError("SS2Dm0 requires d_inner to be divisible by dt_rank")

        self.forward_core = partial(
            self.forward_corem0, force_fp32=False, dstate=self.d_state
        )
        projected_dim = self.d_inner if self.disable_z else 2 * self.d_inner
        self.in_proj = nn.Linear(self.d_model, projected_dim, bias=bias)
        self.act = act_layer()
        self.with_dconv = d_conv > 1
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                self.d_inner,
                self.d_inner,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                groups=self.d_inner,
                bias=conv_bias,
            )

        self.x_proj_weight = nn.Parameter(
            torch.stack(
                [
                    nn.Linear(
                        self.d_inner, self.dt_rank + 2 * self.d_state, bias=False
                    ).weight
                    for _ in range(4)
                ],
                dim=0,
            )
        )
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if initialize != "v2":
            raise ValueError("The UCPMamba Mamba2 route requires SS2Dm0 init v2")
        head_dim = self.d_inner // self.dt_rank
        self.Ds = nn.Parameter(torch.ones((4, self.dt_rank, head_dim)))
        self.A_logs = nn.Parameter(torch.zeros((4, self.dt_rank)))
        self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((4, self.dt_rank)))

        self.initial_state = None
        if with_initial_state:
            self.initial_state = nn.Parameter(
                torch.zeros((1, 4 * self.dt_rank, head_dim, self.d_state)),
                requires_grad=False,
            )

    def forward_corem0(
        self,
        x: torch.Tensor,
        force_fp32=False,
        chunk_size=64,
        dstate=64,
        selective_scan_backend=None,
        **kwargs,
    ) -> torch.Tensor:
        """Cross-scan a feature map and execute Mamba2's chunked SSD scan."""
        del kwargs
        batch, height, width, inner_dim = x.shape
        groups, heads_per_group = self.A_logs.shape
        _, _, head_dim = self.Ds.shape
        if inner_dim != heads_per_group * head_dim or dstate != self.d_state:
            raise ValueError("Inconsistent SS2Dm0 state/head dimensions")

        length = height * width
        xs = _cross_scan(x)
        x_dbl = torch.einsum("b l k d, k c d -> b l k c", xs, self.x_proj_weight)
        dts, bs, cs = torch.split(
            x_dbl, [heads_per_group, dstate, dstate], dim=3
        )
        xs = xs.contiguous().view(
            batch, length, groups * heads_per_group, head_dim
        )
        dts = dts.contiguous().view(batch, length, groups * heads_per_group)
        bs = bs.contiguous().view(batch, length, groups, dstate)
        cs = cs.contiguous().view(batch, length, groups, dstate)
        if force_fp32:
            xs, dts, bs, cs = (tensor.float() for tensor in (xs, dts, bs, cs))

        initial_state = None
        if self.initial_state is not None:
            initial_state = self.initial_state.detach().repeat(batch, 1, 1, 1)
        ys, final_state = selective_scan_chunk_fn(
            xs,
            dts,
            -self.A_logs.float().exp().view(-1),
            bs,
            cs,
            chunk_size=chunk_size,
            d=self.Ds.float().view(groups * heads_per_group, head_dim),
            dt_bias=self.dt_projs_bias.view(-1),
            initial_states=initial_state,
            dt_softplus=True,
            return_final_states=True,
            backend=selective_scan_backend,
        )
        if self.initial_state is not None:
            with torch.no_grad():
                self.initial_state.copy_(final_state.detach().sum(0, keepdim=True))
        y = ys.reshape(batch, length, groups, inner_dim)
        y = _cross_merge(y, height, width)
        return self.out_norm(y).to(x.dtype)

    def forwardm0(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=-1)
            if not self.disable_z_act:
                z = self.act(z)
        if self.with_dconv:
            x = self.conv2d(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        y = self.out_act(self.forward_core(self.act(x)))
        if not self.disable_z:
            y = y * z
        return self.dropout(self.out_proj(y))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"SS2Dm0 expects a 4-D feature map, got {x.shape}")
        if self.channel_first:
            x = x.permute(0, 2, 3, 1)
            return self.forwardm0(x).permute(0, 3, 1, 2).contiguous()
        return self.forwardm0(x)
