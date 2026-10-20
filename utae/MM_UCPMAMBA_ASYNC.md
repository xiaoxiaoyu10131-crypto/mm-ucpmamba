# MMUCPMambaAsync 实验说明

以 `mm-ucpmamba108.sh` 对应的 `MMUCPMambaMid` 为基线，新增独立模型
`mm_ucpmamba_async`。108 启动脚本、既有参数默认值及 MM-UTAE 路线保留。

三路浅层编码 → 动态窗口表征 → 真实时差邻窗交互 → Concat/1×1 Conv →
共享 Down3 → 一个 CSM-ConvLSTM → 共享解码器。浅层两尺度各保留三路
P-MTSA，融合尺度再用一个，总共七个。CSM 使用现有候选状态 g 的 Mamba1。

## 固定首轮配置

| 项目 | 值 |
|---|---|
| 数据与日期 | PASTIS-R，S2/S1A/S1D，2018-09-01 |
| 公共窗口 | `[-200,600)`，24 窗，左闭右开 |
| 模块 A | 均值 + 窗宽归一化斜率 + 五通道观测统计；69→16→64 |
| 趋势条件 | 至少两个不同日期，跨度 ≥2 天 |
| 模块 B | 其他两模态的 k−1/k/k+1，Q/K/V=16，辅助 MLP=11→8→1 |
| 日期惩罚 | 六个有向模态对分别学习 softplus 系数 |
| 残差 | 每模块每模态一个系数，通用初始化之后重置为 0 |
| 编码/解码宽度 | `[64,64,64,128]` / `[32,32,64,128]` |
| 归一化/输出头 | 编码和上采样 GroupNorm 8 组；输出头沿用108，末端 BN=ones |
| batch / 累积 / 精度 | 4 / 1 / FP32 |
| 优化 | Adam，lr=0.001，100轮，固定学习率 |
| 时序 dropout | S2=0.4，S1A=0.2，S1D=0.2；整模态 dropout=0 |
| 种子/选模 | 首轮1，最佳验证 mIoU；后续独立种子2、3 |
| 损失/Void | 沿用当前加权交叉熵，Void=19；评分沿用基线口径 |

窗口日期为 FP64，特征统计至少 FP32；无效值在乘法前用 where 排除。
动态统计按窗口选择相关观测；交互只保留窄通道 Q/K/V，逐窗计算候选。
缺测查询始终无效，没有有效候选时保留原查询，空窗口不能产生伪观测。
Async collate 同步排序影像、日期及显式掩码，再复用现有 padding。
标准数据路径没有云质量标签，观测统计不能称作云质量评价。

实测新增参数 A=6,627、B=12,738，合计19,365；默认总参数2,957,072。

## 正确性与真实 CUDA 检查

在 Mamba1 环境、工程的 utae 目录运行。先检查真实长序列 batch 4：

```bash
python validate_mm_ucpmamba_async.py \
  --dataset-folder /path/to/PASTIS-R \
  --batch-size 4 --patches 8 --steps 2 \
  --output outputs/async_batch4.json
```

默认从测试折2对应的训练折3/4/5中，按实际总观测数选择最长的patch；
不截断帧、不做时序丢帧。这比正式训练的丢帧条件更重。
脚本使用真实 CSM/Mamba，检查第二步新模块梯度、优化更新、checkpoint
参数精确恢复及 FP32 输出容差，保存 GPU、实际长度、显存和固定batch推理耗时。
日期与模块配置变化会拒绝载入 checkpoint。

再使用8–16个训练patch做小样本学习检查，例如8个patch循环多次：

```bash
python validate_mm_ucpmamba_async.py \
  --dataset-folder /path/to/PASTIS-R \
  --batch-size 4 --patches 8 --steps 40 \
  --output outputs/async_small_fit.json
```

查看逐步损失和各循环平均损失；短程下降只用于判断实现能否学习。
这一步结束后的权重不会用于正式训练。小样本验证不能替代五折测试。

## 正式五折

确认上述检查后运行；脚本本身不向服务器提交作业。

```bash
DATASET_FOLDER=/path/to/PASTIS-R \
RES_DIR=/path/to/results/mm_ucpmamba_async_1016/seed_1 \
bash run_mm_ucpmamba_async.sh
```

脚本默认种子1、官方测试折1→5，全部按上表配置从头训练。
`mm-ucpmamba-async.sh` 为 Slurm 包装，沿用 `uvmrnn` 环境，可在服务器上
设置上述环境变量后提交。路径按脚本自身目录定位。

| 测试 | 训练 | 验证 |
|---|---|---|
| 1 | 2,3,4 | 5 |
| 2 | 3,4,5 | 1 |
| 3 | 4,5,1 | 2 |
| 4 | 5,1,2 | 3 |
| 5 | 1,2,3 | 4 |

配置锁定后，对种子2、3分别使用新目录：

```bash
SEED=2 DATASET_FOLDER=/path/to/PASTIS-R \
RES_DIR=/path/to/results/mm_ucpmamba_async_1016/seed_2 \
bash run_mm_ucpmamba_async.sh
```

`FOLDS="1 4 5"` 可指定子集；只在本次运行包含完整五折时自动汇总。
启动器拒绝覆盖已有实验根配置或目标折目录。
修改模型/训练配置时使用新的实验编号和结果目录。

## 配置与保存

独立 Python 入口为 `run_mm_ucpmamba_async.py`，复用现有训练参数。
新增参数：`--async-radius`、`--async-dim`、`--async-trend-min-span`、
`--async-residual-init`（仅支持0）、`--async-dynamic/--no-async-dynamic`、
`--async-interaction/--no-async-interaction`。首轮保持两个模块均开启。
直接使用 `train_semantic.py --model mm_ucpmamba_async` 时仍需显式传入
108的窗口、归一化和初始化配置；推荐使用独立入口固定这些默认值。
推理入口的新增参数默认为 None，从保存配置恢复。

每折保存：

- `config.json`：完整设置、Git版本、工作区状态及源码SHA256。
- `model.pth.tar`：最佳验证 checkpoint，附优化器、配置和模型元数据。
- `last.pth.tar`：最后一轮 checkpoint、优化器和Torch随机数状态。
- `trainlog.json`：训练/验证损失、mIoU、学习率、耗时、实际长度及GPU成本。
- `test_metrics.json`、`conf_mat.pkl`、`class_metrics.json`：完整矩阵、各类
  IoU/Precision/Recall/F1及像素数，9项总体指标。
- 模块诊断：趋势有效像素窗口数、有效观测数、实际时差一/二阶和、时差直方图、
  同箱/邻箱权重总和及每轮末残差系数。比例由分子/分母计算。

时差直方图按 `|Δt|/窗宽` 划分，边界为0.25、0.5、1、2、3。
测试日志另存同一实际batch预热1次、计时3次的推理耗时及输入尺寸。
保留完整混淆矩阵；评分时按108口径排除Void行列，不改变既有评分规则。
`class_metrics.json` 的分数为0–1；原训练日志的mIoU/OA沿用百分数。

完整五折后生成 `async_summary.json`，分别列出五折等权平均和合并混淆矩阵指标。
三个种子训练完成后运行：

```bash
python -m src.mm_async_results /path/to/seed_1 /path/to/seed_2 /path/to/seed_3
```

输出并保存 `async_multiseed_summary.json`，记录两种汇总口径的均值及样本标准差。
配置不同或种子重复会拒绝合并。

## 当前验证边界

2026-10-16：Windows CPU的模块/108相关回归103项通过；WSL原有MM-UTAE
回归43项通过，Async的训练保存推理与启动器测试2项通过。
新增实现通过pyflakes、mypy及语法检查。

真实 PASTIS-R patch10005，未截断S2/S1A/S1D=43/65/70帧，在WSL
uvmrnn、PyTorch2.3.1+cu118、GTX1660SUPER上，默认宽度、物理batch1
完成真实Mamba前向、两次反向/Adam更新及checkpoint恢复。
峰值已分配4,637.5 MiB，峰值预留5,092 MiB；两次损失3.2283→3.1770。
该结果仅证明这个batch能运行，不证明batch4或五折性能。

另用8个真实训练patch、完整43/65/70帧、batch1循环两轮，共16次更新。
两轮平均训练损失3.1247→2.9126；各主要模块梯度有限且非零，保存参数
逐值恢复、恢复输出一致。峰值已分配4,638.2 MiB，固定batch推理约1,299 ms。
这是短程学习检查，尚未达到充分拟合，也不代表研究性能。
报告位于 `outputs/mm_ucpmamba_async_small_fit.json`。

batch4最长序列测试在本机6GB GPU运行约10分钟仍未完成第一次更新，
设备显存使用接近5.8GiB，已停止本轮测试进程。没有捕获到CUDA OOM异常，
因此记为“未完成验证”；不能据此声称batch4通过或测得其峰值已分配显存。
报告位于 `outputs/mm_ucpmamba_async_cuda_batch4.json`，需服务器复验。
正式五折和三种子实验均未启动。
