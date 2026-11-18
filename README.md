# EACT-VFI

这是方案 A 的完整实现，全称 **Event-Adaptive Continuous Trajectories with Brightness-Consistent Splatting for Arbitrary-Time Event-based VFI**。代码目录按 OF-AF 的方式组织成 `models / datasets / util`，训练入口是 `train_model.py` / `train_model.sh`，测试入口是 `test_model.py`。

| 创新点 | 作用 | 代码位置 | 开关 |
| --- | --- | --- | --- |
| A1 EATP 事件密度自适应分段 | 按累计事件量定分段边界；BiFEB 每一步输入段长 | `util/eatp.py`、`util/voxelization.py`、`models/BiFEB.py`（`dt` 条件） | `--partition eatp`、`--time_cond true` |
| A2 CTQ 连续时间查询 | 修正后的线性查询，加上只作用于两条后向流的事件残差；残差乘 4λ(1−λ)，在锚点处精确为 0 | `models/CTQ.py` | `--query linear --query_res bwd`（Hermite 单独消融：`--query hermite`） |
| A3 EBC-Splat 事件亮度一致性度量 | 由事件积分出目标时刻亮度，作为 softmax splatting 的度量；修复掩码与 splat 方式无关 | `models/ebc.py`、`models/BiW.py` | `--splat ebc --mask_mode geo`（另有 `avg`、`niklaus`） |

## 0. 修订记录（针对三条评审意见）

三条意见都成立，已按建议修改：

1. **CTQ 残差破坏锚点约束**：旧版 `hermite_res` 的残差没有窗口，锚点左侧看到的是上一段的全部事件，右侧看到的是下一段的全部事件，训练后锚点两侧会跳变。现在残差乘 w(λ)=4λ(1−λ)（`--res_window bump`），或 16λ²(1−λ)²（`bump2`，配合 Hermite 保留 C1）。默认只修正 F_{t→0}、F_{t→1}，Hermite 只用在前向流上并单独消融。冒烟测试用非零残差权重检查：锚点处取值精确、两侧无跳变，并与去掉窗口时的跳变对比。
2. **时间轴**：
   - 默认 `--time_axis frame`：所有裁剪共用同一对关键帧的真实时刻，窗口外的事件直接丢弃，不再截断到 0/1；
   - GoPro 的时间戳查找加了误差检查；
   - BS-ERGB 可用 `--bsergb_ts_file` 显式指定时间戳文件（如 `images_timestamps.txt`）；
   - 单位换算用 `--event_time_scale` / `--frame_time_scale`；
   - `--time_axis legacy` 只留给旧基线复现，取未裁剪事件的 min/max，与裁剪无关；
   - A2/A3 用到的段内事件和事件积分，改为由数据集按原始时间戳在查询时刻精确切分（`--query_events exact`），计数不会因正负抵消而丢失。
3. **EBC 软空洞掩码**：删掉了由 Σexp(Z) 得到的空洞（单源 Z=−2 时空洞 0.865，多源碰撞时为 0）。`--mask_mode` 与 `--splat` 现在相互独立：
   - `ofaf`：忠实复现 OF-AF，二值空洞 + 0.1 硬阈值；
   - `geo`：几何覆盖 1−min(Σ 双线性权重, 1)，加可学习阈值的 sigmoid 差异掩码；`avg`、`ebc`、`niklaus` 三种 splat 共用这一套掩码，对比时只有 splat 度量不同。
   - 训练时 `--grad_report N` 记录前向流头、C、α、diff_tau、CTQ 残差的梯度范数；测试时 `--splat_diag true` 统计多对一碰撞区的投影误差。

顺带更正一处旧文档的错误：OF-AF 的 `flow1t` 和 `Query` 权重方向相反，**在 pct=0 时也有影响**。这时 `flowt0`、`flowt1`、`ctxt0`、`ctxt1`、`flow1t` 取的是锚点 k+1，只有 `flow0t` 取锚点 k。它与训练时的约定一致，网络可以学会补偿（BS-ERGB 只训练 pct=0，可完全补偿；GoPro skip 31 训练同时有 pct 0 和 0.5，只能折中）。pct=0.5 时两种权重结果相同；其它 pct 时，后向流在段内按时间倒序变化，并在段边界跳过两个锚点。

## 目录

```
EACT-VFI/
├── models/      blocks, softsplat(纯 PyTorch), encoder, BiFEB, CTQ, ebc, BiW, model, loss
├── datasets/    base(样本构建/增强/精确查询事件), gopro, bsergb
├── util/        event(读取/裁剪), voxelization(体素与查询时刻精确切分), eatp, metrics, misc
├── options.py   训练和测试共用的参数
├── evaluator.py 评测循环(网格/非网格拆分, 密集插帧时序指标, splat 诊断)
├── train_model.py / train_model.sh
├── test_model.py
├── tools/check_time_alignment.py  事件与帧时间戳是否对齐(先跑)
├── tools/check_ofaf_grads.py      哪些参数真的拿到梯度(本仓库 / OF-AF 官方代码)
├── tools/eval_ofaf_offgrid.py     用 OF-AF 官方代码和权重做非网格评测
└── tests/       smoke_test.py(PyTorch), util_test.py, util_test_time.py(numpy)
```

## 1. 先跑测试

```bash
pip install -r requirements.txt
python tests/util_test.py         # numpy：EATP、体素、指标
python tests/util_test_time.py    # numpy：时间轴、窗口外事件、精确切分、计数
python tests/smoke_test.py        # PyTorch，只用 CPU，约 1–2 分钟
```

`smoke_test.py` 检查以下内容：

- `softsplat` 与逐像素朴素实现一致，并通过 gradcheck；
- CTQ：
  - 基础插值精确还原锚点、精确重现线性运动；`ofaf` 模式与 OF-AF 发布代码的公式一致；
  - 残差权重设为非零后，锚点处取值仍精确，两侧跳变 ≈0（去掉窗口时有明显跳变）；
  - `bump2` + Hermite 在锚点两侧的斜率一致；
  - `--query_res bwd` 不改变前向流；
- 事件积分：近似和精确两条路径都守恒，精确路径给出真实事件数；
- 掩码：几何覆盖对光流可导，OF-AF 的 avg 空洞几乎不可导；
- 11 种配置 ×（均匀、EATP）边界 ×（近似、精确事件）都能完成前向和反向传播；
- **梯度验收**（float64）：
  - 忠实 OF-AF 配置下，前向流头梯度 < 1e-4 × 后向流头梯度；
  - `geo` 掩码下前向流头梯度明显非零；
  - `ebc` + `geo` 时 α、C、diff_tau 都有梯度；`ebc` + `ofaf` 掩码时 α、C 梯度为 0；
- 合成的 GoPro/BS-ERGB 目录：真实帧时间轴、精确查询特征、批处理，以及含 splat 诊断的评测流程。

> 交付环境没有 PyTorch。numpy 测试已实测通过；PyTorch 部分做了编译检查、未定义名检查，所有训练配置都用真实参数解析器解析过。关键数值结论（锚点跳变、掩码可导性、旧空洞值）另用 numpy 独立复算过。请先跑 `smoke_test.py`，有报错把输出贴给我。

## 2. 数据与时间轴

目录结构与 OF-AF 相同：

```
GOPRO/{train,test}/<scene>/imgs/*.png
GOPRO/{train,test}/<scene>/events_clean_8x_v2/*.npz          # x, y, t, p
GOPRO/{train,test}/<scene>/timestamps_clean_8x_v2.txt
BSERGB/{3_TRAINING,1_TEST}/<scene>/images/*.png
BSERGB/{3_TRAINING,1_TEST}/<scene>/events/*.npz              # x, y, timestamp, polarity
BSERGB/.../<scene>/images_timestamps.txt 或 images/timestamp.txt   # 可选，帧时间戳
```

**第一步先核对时间对齐**：

```bash
python tools/check_time_alignment.py --dataset bsergb --root /DATASSD1/BSERGB/1_TEST --bsergb_ts_file images_timestamps.txt
python tools/check_time_alignment.py --dataset gopro  --root /DATASSD1/GOPRO/test
```

它逐个窗口报告事件时间范围与关键帧区间的重合比例。比例低通常意味着单位不一致（μs 与 s）、存在偏移，或者事件文件与帧的对应关系错了。可以用 `--event_time_scale` / `--frame_time_scale` 修正。

- `--time_axis frame`（默认，A1–A3 的物理时间实验必须用它）：窗口两端取关键帧真实时刻，目标时刻 = (ts_j − ts_0)/(ts_1 − ts_0)，窗口外事件丢弃。重合比例低于 90% 时警告，低于 50% 时报错。BS-ERGB 找不到时间戳文件时退回 legacy，并给出警告。
- `--time_axis legacy`：目标时刻 j/(skip+1)，事件轴取**未裁剪**窗口事件的 min/max。OF-AF 训练时用的是裁剪后事件的 min/max，结果依赖裁剪位置，这一点没有复现。
- `--query_events exact`（默认）：每个查询时刻 t 都在原始时间戳上切分，得到三类量：
  - 该段 t 之前的事件（落在该段自己的 bin 网格上），t 之后的部分由该段体素相减得到；
  - S_{0→t} 与真实事件数 A_{0→t}；
  - 全窗口的总和。
  
  `approx` 是按三角核把 bin 拆分的旧近似，无法恢复已抵消的正负事件，也拿不到精确计数。
- **极性编码**：默认 `--polarity auto`，检测到 {0,1} 时自动映射为 ±1，并给出警告。

## 3. OF-AF 官方权重的非网格评测

```bash
python tools/eval_ofaf_offgrid.py --ofaf_root /path/to/OF-AF \
    --checkpoint /path/to/OF-AF/train/bsergb.pth \
    --dataset bsergb --data_root /DATASSD1/BSERGB --skips 1,2,3,4,5,6
python tools/check_ofaf_grads.py --impl ofaf --ofaf_root /path/to/OF-AF   # 前向流头的梯度
```

第一个脚本完全使用 OF-AF 自己的代码，输出每帧 CSV（含 `grid_frac`，0 表示正好在 1/16 网格点上）。n=16 时，skip 为 1、3、7、15 的目标全部在网格点上；skip 为 2、4、5、6 时大多不在网格点上。

注意：这里测到的非网格掉点，一部分来自上面提到的 OF-AF 权重方向问题。论文里应表述为“分段线性查询在非网格时刻不具时间一致性”。

## 4. 训练

```bash
bash train_model.sh a0fix          # 修正后的基线
bash train_model.sh full           # A1 + A2 + A3
```

每一行只比它的对照行多改一处：

| 配置 | 时间轴 | 分段 | 查询 | 残差 | splat | 掩码 | 对照 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `a0` | legacy | 均匀 | ofaf | – | avg | ofaf | OF-AF 复现 |
| `a0lin` | legacy | 均匀 | linear | – | avg | ofaf | a0（查询权重） |
| `a0frame` | frame | 均匀 | linear | – | avg | ofaf | a0lin（时间轴） |
| `a0fix` | frame | 均匀 | linear | – | avg | geo | a0frame（掩码） |
| `a1` / `a1_n8` | frame | EATP（n=16/8） | linear | – | avg | geo | a0fix |
| `a2h` | frame | 均匀 | Hermite（仅前向流） | – | avg | geo | a0fix |
| `a2r` | frame | 均匀 | linear | bwd, 4λ(1−λ) | avg | geo | a0fix |
| `a2r2` | frame | 均匀 | Hermite（仅前向流） | bwd, 16λ²(1−λ)² | avg | geo | a2h |
| `a3n` | frame | 均匀 | linear | – | niklaus | geo | a0fix |
| `a3` / `a3c` | frame | 均匀 | linear | – | ebc（C 固定 / 可学习） | geo | a0fix、a3n |
| `full` | frame | EATP | linear | bwd | ebc | geo | — |

- 最少需要跑：`a0`、`a0fix`、`a1`、`a2r`、`a3`、`full`。
- 默认全部按 OF-AF 训练方式：GoPro skip 31，每个窗口 31 个目标。`*_mix` 版本混合 skip∈{3,4,5,6,7,9,11,15}，让残差见到更多段内位置。
- `--grad_report 200`（`train_model.sh` 默认开启）会每 200 次迭代往 `train_log.txt` 写一行梯度范数，包括 `fwd_flow_head`、`bwd_flow_head`、`ctq_residual`、`C`、`beta`、`alpha`、`diff_tau`。验收看三点：
  - `geo` 下 `fwd_flow_head` 不再≈0；
  - `ebc` 下 `alpha`、`C` 持续非零；
  - 训练结束时 `diff_tau` 没有漂到让差异掩码全 0 或全 1 的位置。

## 5. 测试与验收协议

```bash
# OF-AF 协议（GoPro 只评 j=2,4,6,…；BS-ERGB 评全部帧）
python test_model.py --checkpoint train/full/epoch19.pth --datasets gopro,bsergb --skips 7,15
# A2 的直接收益：GoPro skip 31 的奇数帧，t=j/32 正好落在两个 1/16 锚点中间
python test_model.py --checkpoint train/a2r/epoch19.pth --datasets gopro --skips 31 --eval_frames odd
# A2 的非回退检查：BS-ERGB skip 1/3 的目标都在网格点上（w=0），应与 a0fix 持平
python test_model.py --checkpoint train/a2r/epoch19.pth --datasets bsergb --skips 1,3
# A3：多对一碰撞区的前向投影误差
python test_model.py --checkpoint train/a3/epoch19.pth --datasets gopro --skips 7 --splat_diag true
# 时序一致性：每个窗口额外渲染 63 帧，计算相邻帧的 warping error（不需要 GT）
python test_model.py --checkpoint ... --datasets bsergb --skips 3 --dense 63 --max_windows 50
```

- 结果写到 `eval/eact/<dataset>/<skip>/`：`frames.csv` 是逐帧结果，`summary.txt` 汇总了整体、网格时刻和非网格时刻三组 PSNR。
- 用 `--splat_diag true` 时，CSV 还会多出三列：
  - `coll_frac`：碰撞像素占比，即覆盖度 > `--collision_thr`（默认 1.5）的像素；
  - `err_coll`：碰撞区内 |I_{src→t} − I_t|×255 的均值；
  - `err_single`：单源区内的同一误差。
- A3 的判断标准：`a3` 相对 `a0fix` 和 `a3n`，`err_coll` 下降、`err_single` 基本不变。
- 模型参数、分段参数以及 `time_axis`、`query_events` 默认从 checkpoint 恢复。
- PSNR/SSIM 的 uint8 转换沿用 OF-AF 的截断方式。时序指标里的光流来自 OpenCV DIS，对比时要注明。

## 6. 实现细节与和 OF-AF 的差异

- **EATP**：边界为 t_k = λ·k/n + (1−λ)·N⁻¹(k/n)，N 是混入均匀先验 γ 后的累计事件曲线。每段最短为 λ/n，最长为 `--eatp_dmax`/n，两个约束都由构造保证。可选 `--eatp_mode dense`，只统计事件最密的区块。
- **CTQ**：
  - 用 searchsorted 定位所在段 k 和段内位置 λ；
  - 基础插值可选 `ofaf`（复现）、`linear`（方向一致）或 `hermite`（非均匀 Catmull-Rom，默认只用于前向流）；
  - 残差网络的输入：t 前、后两部分子段事件，λ 的傅里叶编码，段长，当前四条流，以及上下文；
  - 残差输出零初始化，再乘窗口 w(λ)；
  - 用均匀 16 段时，GoPro skip 7/15 和 BS-ERGB skip 1/3 的目标都在锚点上，w=0，所以 A2 只在非网格时刻直接起作用。
- **EBC-Splat**：
  - 目标亮度有两个估计，L̂ₜ = L₀ + C·S₀→ₜ 和 L₁ − C·Sₜ→₁，按 softmax(−β·A) 融合，A 是真实事件数；
  - 度量为 Z(p) = −α·ρ(L_src(p) − L̂ₜ(p + F(p)))；
  - Z 只决定碰撞时哪个源像素占优，不参与空洞判断。
- **掩码**：
  - `ofaf` 完全复现 OF-AF，归一化为 s/(s+1e-7)，前向流头只拿到 ~1e-7 量级的梯度，覆盖度极小的像素处偶有尖峰；
  - `geo` 让前向流和 splat 度量通过覆盖度和软差异掩码拿到直接梯度。
- **forward splatting**：用纯 PyTorch 的 `index_add` 实现，不依赖 cupy。
- **其它差异**：
  - BackwardWarp 用 `align_corners=True` 和 border 填充；OF-AF 有半像素偏移，填充方式为 zero；
  - LPIPS 损失输入归一化到 [−1,1]；
  - 这些会让 `a0` 与官方 OF-AF 的数字略有出入，论文里应同时报告两者。

## 7. 注意

- 主干的层配置与 OF-AF 一致，这是为了让对比公平。OF-AF 仓库没有 LICENSE，公开本仓库代码前，应先征得 OF-AF 作者同意。
- 显存：CTQ 残差和精确查询特征都是每个查询时刻全分辨率计算。显存不足时，可以减小 `--targets_per_sample` 或 `--crop`。旧 checkpoint 的 `query=hermite_res` 会自动映射为 Hermite + 全部残差，并加上窗口。
