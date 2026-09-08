"""Minimal SSD implementation adapted from the official VMamba Mamba2 path.

Reference:
https://github.com/MzeroMiko/VMamba/blob/main/classification/models/mamba2/ssd_minimal.py
"""

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


def segsum(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable lower-triangular segment sums."""
    length = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=length)
    mask = torch.tril(
        torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=-1
    )
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(
        torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=0
    )
    return x_segsum.masked_fill(~mask, -torch.inf)


def ssd_minimal_discrete(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    block_len: int,
    initial_states: torch.Tensor | None = None,
):
    """SSD block decomposition from Mamba-2 Listing 1."""
    assert x.dtype == a.dtype == b.dtype == c.dtype
    assert x.shape[1] % block_len == 0
    x, a, b, c = [
        rearrange(tensor, "b (c l) ... -> b c l ...", l=block_len)
        for tensor in (x, a, b, c)
    ]
    a = rearrange(a, "b c l h -> b h c l")
    a_cumsum = torch.cumsum(a, dim=-1)

    decay = torch.exp(segsum(a))
    y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", c, b, decay, x)

    decay_states = torch.exp(a_cumsum[:, :, :, -1:] - a_cumsum)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", b, decay_states, x)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(a_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    state_decay_out = torch.exp(a_cumsum)
    y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", c, states, state_decay_out)
    y = rearrange(y_diag + y_off, "b c l h p -> b (c l) h p")
    return y, final_state


def mamba_chunk_scan_combined_torch(
    x,
    dt,
    a,
    b,
    c,
    chunk_size,
    d=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    seq_idx=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    return_final_states=False,
):
    """Pure-PyTorch fallback for the fused Mamba2 SSD chunk scan."""
    del seq_idx, dt_limit
    batch, seqlen, ngroups, dstate = b.shape
    nheads = x.shape[2]
    while seqlen % chunk_size != 0:
        chunk_size >>= 1

    if nheads != ngroups:
        assert nheads % ngroups == 0
        repeat_count = nheads // ngroups
        b = b.view(batch, seqlen, ngroups, 1, dstate).repeat(
            1, 1, 1, repeat_count, 1
        ).view(batch, seqlen, nheads, dstate)
        c = c.view(batch, seqlen, ngroups, 1, dstate).repeat(
            1, 1, 1, repeat_count, 1
        ).view(batch, seqlen, nheads, dstate)

    if dt_bias is not None:
        dt = dt + dt_bias
    if dt_softplus:
        dt = F.softplus(dt)
    u = x * dt.unsqueeze(-1)
    y, state = ssd_minimal_discrete(
        u, a * dt, b, c, block_len=chunk_size, initial_states=initial_states
    )
    if d is not None:
        y = y + d.view(y.shape[-2], -1) * x
    if z is not None:
        y = y * (z * torch.sigmoid(z))
    return (y, state) if return_final_states else y


try:
    from mamba_ssm.ops.triton.ssd_combined import (
        mamba_chunk_scan_combined as _mamba_chunk_scan_combined,
    )
except ImportError:
    _mamba_chunk_scan_combined = None


def selective_scan_chunk_fn(
    x,
    dt,
    a,
    b,
    c,
    chunk_size,
    d=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    seq_idx=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    return_final_states=False,
    backend=None,
):
    """Dispatch to fused Mamba2 SSD on CUDA, otherwise use the SSD reference."""
    use_torch = backend == "torch" or _mamba_chunk_scan_combined is None
    use_torch = use_torch or not x.is_cuda
    if x.is_cuda:
        use_torch = use_torch or torch.cuda.get_device_capability(x.device)[0] < 8
    fn = mamba_chunk_scan_combined_torch if use_torch else _mamba_chunk_scan_combined
    kwargs = dict(
        x=x,
        dt=dt,
        A=a,
        B=b,
        C=c,
        chunk_size=chunk_size,
        D=d,
        z=z,
        dt_bias=dt_bias,
        initial_states=initial_states,
        seq_idx=seq_idx,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
        return_final_states=return_final_states,
    )
    if use_torch:
        kwargs = {
            key.lower() if key in {"A", "B", "C", "D"} else key: value
            for key, value in kwargs.items()
        }
    return fn(**kwargs)
