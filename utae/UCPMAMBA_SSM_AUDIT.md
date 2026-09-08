# UCPMamba SSM 实现审计报告

审计日期：2026-09-07

## 结论

`ucpmamba_vmambav2` 保留原 `SS2Dv2` 配置变体；新增的
`ucpmamba_mamba2` 使用独立 `SS2Dm0`，实际进入 Mamba2 的分块 SSD，二者不再混用。

实现参考官方 VMamba 主分支：

- [`SS2Dm0` / `forward_corem0`](https://github.com/MzeroMiko/VMamba/blob/main/classification/models/vmamba.py)
- [Mamba2 `ssd_minimal.py`](https://github.com/MzeroMiko/VMamba/blob/main/classification/models/mamba2/ssd_minimal.py)
- [`vmambav2_tiny_224.yaml`](https://github.com/MzeroMiko/VMamba/blob/main/classification/configs/vssm/vmambav2_tiny_224.yaml)

## 三个入口的实际路径

| 模型入口 | 实际 SS2D 实现 | core | 底层 scan | state dimension |
|---|---|---|---|---:|
| `ucpmamba` | `SS2D` dispatcher / `SS2Dv2` | `forward_corev2` | `SelectiveScanMamba` | 16 |
| `ucpmamba_vmambav2` | `SS2D` dispatcher / `SS2Dv2` | `forward_corev2` | `SelectiveScanOflex` | 1 |
| `ucpmamba_mamba2` | `SS2Dm0` | `forward_corem0` | `selective_scan_chunk_fn` / SSD | 64 |

`ucpmamba_vmambav2` 使用 `v05_noz, d_state=1, ratio=2.0`。真正 Mamba2
使用官方 Mamba2 示例的 `m0_noz, d_state=64, ratio=1.0, GELU, init=v2`，没有沿用
VMamba v2 的 `d_state=1`。

## 参数量

统计对象为当前语义分割完整模型配置：`encoder_widths=[64,64,64,128]`、
`decoder_widths=[32,32,64,128]`、`out_conv=[32,20]`。

| 模型入口 | 总参数量 | 可训练参数量 |
|---|---:|---:|
| `ucpmamba` | 1,820,451 | 1,820,451 |
| `ucpmamba_vmambav2` | 1,821,155 | 1,821,155 |
| `ucpmamba_mamba2` | 1,839,683 | 1,839,683 |

可用以下命令复算：

```bash
python tests/audit_ucpmamba_ssm.py
```

## 运行时调用栈证据

审计脚本通过包装实际运行函数记录到：

```text
SS2Dm0.forward
  -> SS2Dm0.forward_corem0
    -> selective_scan_chunk_fn
      -> mamba_chunk_scan_combined_torch
        -> ssd_minimal_discrete
```

因此新入口没有进入 `SS2Dv2.forward_corev2`，也没有调用
`SelectiveScanOflex`。在 SM80 及以上 CUDA 设备上，`selective_scan_chunk_fn`
优先分派到 `mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined`；当前
GTX 1660 SUPER 为 SM75，其 Triton 2.3.1 融合内核编译失败，因此实测使用同一
入口下的 PyTorch SSD Listing-1 后端，而不是 VMamba v2 selective scan。

## Shape 与梯度验证

- `SS2Dm0` 核心：随机输入 `[1,64,8,8]`，输出 `[1,64,8,8]`；输入梯度和全部
  Mamba2 核心参数梯度均存在、有限，且存在非零梯度。
- 完整模型：随机输入 `[1,1,10,128,128]`；`ucpmamba` 与
  `ucpmamba_mamba2` 输出均为 `[1,20,128,128]`。
- 完整 `ucpmamba_mamba2` 已完成 CUDA forward/backward；输出有限，所有已产生
  的参数梯度有限，`beta.grad` 存在。
- `beta` 初值仍为 0，所以完整模型第一次反向时 VSS 内部梯度受 `beta` 门控为 0；
  独立核心反向测试证明 SSD 本身梯度链正常。这保持了原 `g_only + beta` 设计。

测试命令与结果：

```bash
pytest -q tests/test_ucpmamba_ssm_variants.py
# 6 passed
```

## 修改边界

- U-Net 编码器、普通 ConvLSTM 门计算、`g_only`、`beta`、跨帧
  `PhenologyAttentionV2`、decoder、数据集、loss 和训练执行流程均未改变。
- `train_semantic.py` 与 `test_semantic.py` 只增加/重命名模型名称选项。
- 测试比较了三个入口传给 `UCPMamba` 的非时序配置，并验证所有 VSS 之外的
  state-dict 参数名和 shape 完全一致。
- 工作区原有的 `src/dataset.py`、`src/learning/weight_init.py` 未由本次任务修改。
