# MM-UCPMamba 的独立 mid fusion 接口

新模型名称为 `mm_ucpmamba_mid`，原 `mm_ucpmamba` 接口和主干文件保留。
这版是主路在时序编码前融合、浅层跳连在时序聚合后融合的混合结构。

## 数据流

默认 128×128 输入和 `[64,64,64,128]` 编码宽度：

1. S2、S1A、S1D 分别执行 in_conv、Down1、Down2，输出各自
   `[B,T_m,64,32,32]`，保留独立的真实日期和掩码。
2. 在固定公共日期区间内，分别按 K 个时间窗计算逐有效像素均值，
   得到三个 `[B,K,64,32,32]`。默认 K=24，不作跨窗插值。
3. 每个时间窗拼接三个特征，以已有 concat + 1×1 投影压回 64 通道。
4. 融合序列共用 Down3 和一个 CSM-ConvLSTM，得到 16×16 瓶颈。
5. 32×32 跳连使用一个共享 P-MTSA；128×128、64×64 跳连仍分别
   使用 P-MTSA 后融合；三个跳连和瓶颈进入共享 decoder。

仅实现 quality-free concat 基线；`qaf` 和 `--use-quality-temporal`
不能用于新接口。这里没有加入双向门控或额外 SAR 物理特征。

## 训练和推理入口

在原训练命令上选择新模型，并加入：

```text
--model mm_ucpmamba_mid --sar-orbits both --fusion-mode concat
--mid-num-bins 24 --mid-start-day START --mid-end-day END
```

START、END 是相对 `--ref_date` 的天数，公共区间为左闭右开
`[START, END)`。起止日必须显式提供，所有样本、训练和推理共用同一
日历区间。应依据实际物候季和数据日期设定，不能按每批样本动态变化。
有效观测超出区间时会报错，避免静默丢帧；若要只用特定物候季，
应先明确筛选观测，再配置对应边界。

例如现有 PASTIS 加载器日期表覆盖 `[-200,600)`，可用这个区间检查
完整数据链；它覆盖约 800 天，不是四川单个物候季的推荐设置。

默认结果目录会包含 `mm_ucpmamba_mid_s2_s1a_s1d`，不会复用原模型的
默认结果目录。显式 `--res_dir` 仍由调用者指定，建议使用独立目录。
`conf.json` 和 checkpoint metadata 保存时间窗设置；测试入口从保存
配置恢复参数。加载 checkpoint 时，窗口数或日历边界不同会明确拒绝。

新模型支持单轨 `--sar-orbits S1A` 或 `S1D`。新旧模型的权重结构不同，
新接口从头训练，不把原模型 checkpoint 当作兼容权重加载。

## 直接调用和显式掩码

```python
from src.backbones.mm_ucpmamba_mid import MMUCPMambaMid

model = MMUCPMambaMid(num_bins=24, start_day=start, end_day=end)
logits = model(
    x_s2, x_s1a, x_s1d,
    batch_positions_s2=dates_s2,
    batch_positions_s1a=dates_s1a,
    batch_positions_s1d=dates_s1d,
    pad_masks=pad_masks,
    valid_masks=valid_masks,
)
```

- 影像 `[B,T_m,C_m,H,W]`，日期 `[B,T_m]`，各模态长度可以不同。
- `pad_masks` 是以 S2/S1A/S1D 为键的字典，值为 bool `[B,T_m]`，
  True 表示 padding。训练和测试的新 collate 根据实际长度自动生成。
- `valid_masks` 是相同键的字典，值为 bool `[B,T_m]` 或
  `[B,T_m,1,H,W]`，True 表示有效，可表达云、阴影等缺测。
- 模型始终先排除 padding，再应用有效性掩码。无效值中的 NaN/Inf
  不会作为观测参与编码和均值；有效日期必须有限且在公共区间内。
- 标准 PASTIS 加载器没有提供云/SCL 掩码，因此默认只识别 padding，
  不自动判云。自定义加载器可在 data_dict 中附带 `valid_masks` 和
  `pad_masks` 字典，训练路由会将它们传给新模型。
- 空窗口及缺失模态的特征严格为零；融合后全空位置也清除卷积偏置。
  所有模态都缺失的时间窗不会更新该样本的时序记忆。
- 有效性下采样采用单元内“任一像素有效”，不会丢掉孤立有效像素。
  低尺度 counts 表示对应特征单元的有效帧数，不是原始像素面积。
- 模型直接调用时允许整路 None 或 T=0；至少提供一路张量以确定
  batch 和空间尺寸。没有显式 padding 掩码时，沿用输入全 pad_value
  的兼容推断；有效的全零观测应显式传入 False padding 掩码。
- 训练 batch 保留已选择的模态键；整路缺测用 T=0 张量及空日期，
  或全 False 的有效掩码表示。不同样本可省略各自的可选掩码键。
- 时序状态跳过以样本／时间窗为单位；局部像素缺测不等同于逐像素
  冻结记忆，仍可能从空间邻域和已有状态接收信息。

`return_debug=True` 返回 logits、各模态 aligned_valid/aligned_counts、
共同时间窗和时序 pad_mask、fused_sequence、各阶段形状等信息。
全样本没有观测时，网络仍可输出有限的分类先验；有效性需从掩码判断。

## 验证范围

### batch 4 的可选省显存执行

启动命令加入 `--mid-low-memory`，或直接构造 `MMUCPMambaMid(low_memory=True, ...)`：

- 仅在训练且开启梯度时，对三个浅层空间分支使用非重入 checkpoint。
  原始输入不需要 `requires_grad=True`，分支参数仍正常反传。
- 有效空间帧每 8 帧执行一次卷积；P-MTSA 每 8 个时间步执行一次
  Conv3d，保留两侧真实邻域，然后仍在完整 T 上计算 masked softmax。
  加权乘积也分块归约；日期窗口、时序顺序和网络参数结构保持一致。
- 当前要求 `encoder_norm=group`，防止分块或重计算改变 BatchNorm 统计。
  共享 CSM-ConvLSTM、decoder 和输出 BatchNorm 不参与重计算。
- 推理仍可分块，但不做激活重计算。开关不新增可训练参数，已有同模型
  state_dict 可严格加载；推理入口默认保留 conf.json 中保存的开关。
- batch size 仍为 4，每批直接更新；不需要梯度累积。参考
  `mm-ucpmamba108.sh`，文件为 UTF-8 / LF，可用 `sbatch` 提交。
- 若再次反向 OOM，会额外打印本批三个序列的形状、非 padding 帧数、
  GPU 型号，以及设备剩余显存、进程 allocated/reserved 和运行峰值。

重计算以更多运算换取较少的激活保留；分块可能有正常浮点误差。
CPU 的输出／梯度对照及 saved-tensors 统计不能证明真实 GPU 峰值或
batch 4 已经足够，需要服务器运行覆盖长序列批次与验证集。

新增测试覆盖日期对齐、不同序列长度、空窗、云掩码、缺模态、梯度、
日历配置恢复、混合精度和 CLI 路由。CPU 测试使用真实空间模块、
P-MTSA 与 concat，时序替身只存在于测试文件内；生产模型直接复用
现有 CSM-ConvLSTM，没有替身或 GRU 回退。

实际显存下降比例需要在同一 GPU、batch、输入观测长度和精度下测量。
三路浅层高分辨率时序特征仍保留，不能仅按时序模块数量推算显存。
