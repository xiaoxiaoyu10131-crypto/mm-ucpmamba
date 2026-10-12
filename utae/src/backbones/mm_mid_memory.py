"""mid 模型的分块算子与 OOM 日志，保持帧顺序和时间卷积边界。"""

import json

import torch


def chunked_frame_forward(block, flat, selected, channels, size, chunk_size):
    """GroupNorm 空间块按有效帧分块，输出仅分配一次。"""
    first = selected[:chunk_size]
    encoded = block(flat.index_select(0, first))
    output = encoded.new_zeros(flat.shape[0], channels, *size)
    output.index_copy_(0, first, encoded)
    for indices in selected[chunk_size:].split(chunk_size):
        output.index_copy_(0, indices, block(flat.index_select(0, indices)))
    return output


def chunked_phenology_logits(aggregator, safe, chunk_size):
    """时间块带 Conv3d halo；权重归一化仍由调用者在完整 T 上进行。"""
    conv = aggregator.conv3d
    halo = (conv.kernel_size[0] - 1) * conv.dilation[0] // 2
    if conv.stride[0] != 1 or conv.padding[0] != halo:
        raise ValueError(
            "chunked P-MTSA requires centered, length-preserving temporal convolution"
        )
    length = safe.shape[1]
    logits = []
    for start in range(0, length, chunk_size):
        stop = min(start + chunk_size, length)
        left, right = max(0, start - halo), min(length, stop + halo)
        features = conv(safe[:, left:right].permute(0, 2, 1, 3, 4))
        first, last = start - left, stop - left
        features = features[:, :, first:last]
        logits.append(aggregator.attn_net(features).permute(0, 2, 1, 3, 4))
    return torch.cat(logits, dim=1)


def chunked_weighted_sum(safe, weights, chunk_size, dtype):
    """避免一次分配 [B,T,C,H,W] 的加权乘积；低精度仍用 float32 累加。"""
    b, t, c, h, w = safe.shape
    result = safe.new_zeros(b, c, h, w, dtype=dtype)
    for start in range(0, t, chunk_size):
        stop = start + chunk_size
        result = result + (safe[:, start:stop] * weights[:, start:stop]).sum(dim=1, dtype=dtype)
    return result


def report_mid_oom(inputs, device):
    """反向 OOM 后补充本批形状和显存统计；诊断失败不覆盖原异常。"""
    try:
        details = {
            "sequence_shapes": {
                n: list(inputs[n].shape) for n in ("S2", "S1A", "S1D") if n in inputs
            }
        }
        pads = inputs.get("pad_masks", {})
        details["non_padding_frames"] = {
            n: (~p).sum(dim=1).detach().cpu().tolist() for n, p in pads.items()
        }
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(device)
            details.update(
                gpu=torch.cuda.get_device_name(device),
                device_total_mib=total / 2**20,
                device_free_mib=free / 2**20,
                process_allocated_mib=torch.cuda.memory_allocated(device) / 2**20,
                process_reserved_mib=torch.cuda.memory_reserved(device) / 2**20,
                peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20,
            )
        print("MID backward OOM context: " + json.dumps(details), flush=True)
    except Exception as error:
        print(f"MID OOM diagnostics unavailable: {error}", flush=True)
