# MM-UTAE 三模态 late fusion 对比

此实现用于在同一 PASTIS-R 工程中比较独立 `mm_utae` 与现有 late fusion `mm_ucpmamba`。它基于官方 U-TAE 重建多模态路径；没有拿到原作者完整 MM-UTAE 融合训练源码，也没有验证论文 66.3 mIoU 已复现。

## 依据与证据边界

- [多模态论文](https://arxiv.org/html/2112.07558v1)：§3.1、公式 3 定义每模态独立时空编码、沿通道拼接、统一分类；§3.4 的语义分割分类器为两层 CNN。
- [官方 U-TAE 源码](https://github.com/VSainteuf/utae-paps/blob/main/src/backbones/utae.py)：`encoder=True` 仍执行完整空间 decoder，仅跳过分类 `out_conv`；默认 encoder 通道为 `[64,64,64,128]`，decoder 为 `[32,32,64,128]`。
- [官方 PASTIS-R 数据与加载器](https://github.com/VSainteuf/pastis-benchmark)：S2、S1A、S1D 为三个独立模态，沿用官方五折。

论文没有给融合 CNN 的隐藏宽度、卷积核、归一化及末层激活细节。本实现采用明确记录的 `96→32→20` 两层 3×3 CNN。三支完整默认 U-TAE 的参数量与论文 Table 2 的 1.709M 不一致，因此不猜测 SAR 分支宽度以强行对齐；实际参数量由运行日志和 `conf.json` 记录。

## 结构与分类头

```text
S2  [B,To,10,H,W] → 完整 U-TAE → [B,32,H,W]
S1A [B,Ta, 3,H,W] → 完整 U-TAE → [B,32,H,W]
S1D [B,Td, 3,H,W] → 完整 U-TAE → [B,32,H,W]
                     通道拼接 → [B,96,H,W] → 两层 CNN → [B,20,H,W]
```

三个分支保留各自日期轴和 padding mask，允许 `To != Ta != Td`。输入、标签和官方 fold 划分复用本工程的真实 PASTIS-R 数据路径。

每个分支会裁剪本 batch 全体样本都为 padding 的公共尾部，不删除任何样本的有效观测。其时间编码仍沿用官方 LTAE 的时间维 GroupNorm；变长样本与不同长度的其他样本组成 batch 时，归一化统计可能改变，因此不声称任意 batch 组合下预测完全不变。

现有 `mm_ucpmamba` 在多尺度特征拼接后使用共享空间 decoder；本 `mm_utae` 先由各自完整 U-TAE 恢复高分辨率，再拼接分类。两者都属于 late fusion，但融合位置和 decoder 数量不同，比较结果包含完整模型设计差异，不能只归因于 Mamba 与注意力。

`--mm-utae-head legacy` 保留官方基础 U-TAE 的分类 `ConvBlock` 风格，包括末层 BatchNorm/ReLU；`--mm-utae-head logits` 的末层输出无 BN/ReLU 的原始 logits。后者是为交叉熵采用的明确实现选择，论文未提供足够代码证据确认其原始末层。

## 两种运行配置

| 配置 | 用途 | MM-UTAE 头 | auxiliary 权重 | final BN 初始化 |
|---|---|---|---:|---|
| `comparison`（默认） | 与本工程 `mm_ucpmamba` 共用训练设置 | `legacy` | 0 | `ones` |
| `paper` | 论文 late fusion 的增强训练协议重建 | `logits` | 每模态 0.5 | `legacy` |

两种配置均使用 S2/S1A/S1D、concat、decoder BatchNorm、训练期 temporal dropout `0.4/0.2/0.2`，验证和测试使用全部观测。辅助 loss 为主交叉熵加各模态交叉熵之和乘相应权重；辅助预测只用于训练，不替代主融合预测。

论文 §4.2 / Table 2 的 66.3 对应 auxiliary + temporal dropout；仅 temporal dropout 为 65.8。论文配置为 Adam、学习率 0.001、100 epochs、物理 batch 4、官方五折。`paper` 是依据这些描述重建的协议，不代表精确参数或性能复现。`PROFILE=paper MODEL=mm_ucpmamba` 会被脚本拒绝。

`comparison` 为两模型统一 epoch、学习率、seed、batch、accumulation、dropout、类别、fold 与共享 decoder 的归一化选项。辅助监督为 0；模型本身的编码与融合结构仍不同。若要评价 `paper` 增强的收益，应另做同模型消融，不能把其提升直接归因于骨干网络。

## 训练

在装有依赖的 Linux/WSL 环境中运行。`mm_ucpmamba` 对照还要求该环境能实际执行 Mamba1 CUDA 后端；本脚本不编译后端、不自动激活特定 Conda 环境、不自动提交长期 GPU 作业。

脚本进入自身所在的 `utae` 目录；默认数据目录为 `/share/home/xyxiao/dataset/expduoyuan/PASTIS-R`，启动时检查其存在。所有结果目录都应专用于同一次模型/配置实验。

```bash
# 完整五折 MM-UTAE，默认 comparison。
DATASET_FOLDER=/path/to/PASTIS-R bash run_mm_utae.sh

# 完全相同训练参数重跑现有 late fusion MM-UCPMamba。
MODEL=mm_ucpmamba DATASET_FOLDER=/path/to/PASTIS-R bash run_mm_utae.sh

# 论文增强协议重建，结果放入独立目录。
PROFILE=paper DATASET_FOLDER=/path/to/PASTIS-R bash run_mm_utae.sh

# 指定子集及较小物理 batch。
FOLDS='1 4 5' BATCH_SIZE=2 ACCUMULATION_STEPS=2 \
  RES_DIR=./results/mm_utae_comparison_b2a2 \
  DATASET_FOLDER=/path/to/PASTIS-R bash run_mm_utae.sh
```

默认 `FOLDS='1 2 3 4 5'`、`BATCH_SIZE=4`、`ACCUMULATION_STEPS=1`、`EPOCHS=100`、`LR=0.001`、`SEED=1`、`NUM_WORKERS=8`、`PYTHON=python`、`DEVICE=cuda`。可通过同名环境变量覆盖。默认结果目录为 `./results/${MODEL}_${PROFILE}_s2_s1a_s1d`。

物理 batch 2 + accumulation 2 的有效优化 batch 为 4，但 BatchNorm 按每个物理 batch 统计，不能称其与物理 batch 4 严格等价。两模型对比时应同时使用相同的物理 batch 与 accumulation，并使用独立结果目录记录该设置。

需要 SLURM 时由用户在已配置环境中手工提交；资源和分区按实际服务器要求填写：

```bash
sbatch --gres=gpu:1 --output=mm-utae.log run_mm_utae.sh
```

本工程唯一旧脚本 `mm-ucpmamba108.sh` 当前运行 `mm_ucpmamba_mid`，且使用公共时间分箱与 GroupNorm。此脚本重跑的是 `mm_ucpmamba` late fusion；不能自动把旧 mid 的日志、检查点或指标当成本次 late 对照。

## 保存结果与推理

每折以 `--fold N` 指定官方测试 Fold N，训练/验证划分由现有 `get_fold_split` 决定。类别输出为 20，Void=19 被忽略，指标包含 Background=0，共 19 个有效类。沿用现有指标实现。

每折保存 `Fold_N/model.pth.tar`、`trainlog.json`、`test_metrics.json`、`conf_mat.pkl`；命令和控制台输出保存到 `logs/fold_N_时间_进程号.log`。已有指定折的 `model.pth.tar` 时脚本停止，避免覆盖。默认完整五折结束后复用 `train_semantic.overall_performance` 合并五个混淆矩阵，保存 `overall.json`；该指标是合并混淆矩阵后的全量结果，不等同于五折 mIoU 的算术平均。子集运行不自动五折汇总。

推理从训练结果根目录读取配置和检查点；保持与训练一致的模型、分类头及模态配置：

```bash
python test_semantic.py \
  --dataset_folder /path/to/PASTIS-R \
  --weight_folder ./results/mm_utae_comparison_s2_s1a_s1d \
  --res_dir ./results/mm_utae_comparison_inference \
  --fold 1 --device cuda
```

## 运行验证

```bash
# CPU 结构、IO/损失与已有相关训练逻辑测试；具体覆盖以实际测试文件为准。
python -m pytest -q tests/test_mm_utae.py tests/test_mm_utae_io.py tests/test_mm_utae_launcher.py tests/test_temporal_dropout.py tests/test_gradient_accumulation.py

# 有 CUDA 时执行默认通道和 128x128 输入的真实 GPU 前向/反向及 CPU/GPU 一致性检查。
python -m pytest -q tests/test_mm_utae_cuda.py

# 脚本语法检查，不启动训练。
bash -n run_mm_utae.sh
```

2026-10-14 本地验证记录：新增模型、训练保存推理链、启动脚本及相关旧路由回归共 70 项通过、1 项跳过；默认架构 CUDA 测试 2 项通过。新增模型/配置/验证脚本的 pyflakes、mypy 与语法检查通过。默认 comparison 架构实际参数量为 **3,241,884**。

使用用户真实数据 `F:\PASTIS-R\PASTIS-R`，核验到 2,433 个 patch。在 WSL `uvmrnn`、PyTorch 2.3.1+cu118、GTX 1660 SUPER 上，默认宽度、物理 batch 1 完成：测试 Fold 2 的 patch 10001 全时序 S2/S1A/S1D=43/65/70 帧前向；训练 fold 3/4/5 中 patch 10005 经 temporal dropout 后保留 19/49/50 帧，完成反向和一次 Adam 更新。输出 `[1,20,128,128]`、各支路注意力梯度和全部已产生梯度均有限，融合分类头权重实际更新。没有额外截断时序。该次完整流程峰值已分配显存约 6,024.6 MiB。

复核真实数据的命令：

```bash
python validate_mm_utae.py --dataset-folder /path/to/PASTIS-R --device cuda
```

验证报告默认保存到 `outputs/mm_utae_validation.json`。`--head logits --aux-weight 0.5` 可检查论文增强路线；`--max-frames` 默认 0，表示保留完整时序，正数仅用于显式限制 smoke test 的观测数，报告会记录该限制。

完整训练与推理测试使用 pytest 临时目录里的合成夹具；上述 GPU smoke 使用真实 PASTIS-R。两者都不是五折正式训练结果。服务器物理 batch 4、长时间稳定性和论文 66.3 mIoU 均尚未验证。样本长度和 batch 内 padding 会改变显存；本机 batch 1 的测量不能作为服务器 batch 4 的显存保证。

启动脚本除保护已有检查点外，也会对续跑目录中的 `conf.json` 核验模型与训练配置，拒绝不同配置覆盖同一个结果目录。
