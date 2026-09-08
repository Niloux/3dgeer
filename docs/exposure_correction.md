# PPISP + 局部曝光／色度网格

`use_exposure_correction` 是独立的组合模式，与 `app_opt`、`use_ppisp`、
`use_bilateral_grid` / `use_fused_bilagrid` 互斥。当前支持单 GPU、
`batch_size: 1`、完整图像（`patch_size: null`）。

在已有训练 YAML 中设置以下字段，其余数据、相机、SH、增密等参数保持原值：

```yaml
app_opt: false
use_ppisp: false
use_bilateral_grid: false
use_fused_bilagrid: false
use_exposure_correction: true
bilateral_grid_shape: [16, 16, 8]
exposure_correction_grid_start_iter: 1000
exposure_correction_lr: 0.002
exposure_correction_grid_lr: 0.002
exposure_correction_tv_weight: 10.0
```

也可用 CLI 临时覆盖现有配置，例如：

```bash
source scripts/activate.sh
uv run python examples/simple_trainer.py \
  --config configs/simple_trainer/exhibition_hall.yaml \
  --no-use-bilateral-grid --use-exposure-correction \
  --result-dir results/exhibition_hall_exposure
```

如果基础配置还启用了其他外观模式，需要同时关闭对应开关。
`exposure_correction_grid_start_iter` 使用训练循环的零起始 step；
`steps_scaler` 会一并缩放该值。局部网格的优化器和学习率调度仅在启用后推进，
如果总步数不超过起始 step，则整轮只训练全局部分。

## 计算和约束

```text
原始渲染（含背景）
  → 每帧曝光 + 每物理相机渐晕 + 每帧 PPISP 色度变换
  → 固定恒等 CRF 的 [0,1] 裁剪
  → 每训练图的 (x,y,亮度) 网格：1 个曝光 + 8 个色度参数
  → [0,1] 裁剪 → 图像损失
```

每次有效优化后，全局曝光与色度参数在所有训练帧间做零均值投影；
局部网格在每张图内、每个参数通道上做零均值投影，并施加平方 TV。
曝光值限于 ±16 EV，色度变换只接受非负 RGB，非正输入使用零子梯度。
全局部分保留渐晕中心、正系数和通道差异的正则项，不训练 CRF 或 controller。
左右物理相机共享各自的渐晕参数，每张相机图像有独立曝光／色度和局部网格。

数学实现位于 `examples/lib_exposure_correction.py`，基于 NVIDIA PPISP
的 [Apache-2.0 源码](https://github.com/nv-tlabs/ppisp/tree/df33809f7b3b20ac06de088dfc871b144b8fb54d)，
分工和均值约束参考 LichtFeld Studio 的 `use_exposure_correction`。
组合模式只依赖现有 uv 环境中的 PyTorch，不需要额外的 `ppisp` 包或 CUDA 编译。
独立 `use_ppisp` 模式仍使用原来的外部包。

这是用于效果对照的 PyTorch 实现：网格切片使用 `grid_sample`，
全网格采用普通 Adam 和平方 TV，随后逐图中心化。
它没有移植 LichtFeld 的按图 CUDA 优化器、融合内核或 EXIF 初始化，
因此不保证逐步数值一致或相同的速度／显存占用。

## 保存与评估

- PLY 和 viewer / 轨迹渲染继续使用原始 Gaussian 颜色。
- checkpoint 保存全局参数、相机／帧映射、9 通道网格、格式版本和网格启用状态。
  用同一配置加 `--ckpt <路径>` 加载评估；组合模式与独立 PPISP 的状态不可混用。
  这沿用现有 trainer 的模型快照机制，`--ckpt` 是评估入口，不提供训练续跑。
- `psnr` / `ssim` / `lpips` 是验证集原始渲染指标，`train_` 前缀对应训练集。
- `ppisp_*` / `train_ppisp_*` 是全局 PPISP 结果；验证图只应用所属物理相机的渐晕，
  不估计该图的曝光或色度，不用 GT 拟合。
- `train_exposure_*` 是完整组合结果。网格尚未启用时，它等于全局 PPISP 结果。
  存在天空掩码时，还记录对应 `*_no_sky_*` 指标。
- 训练评估图的 `final` 显示完整组合结果，验证评估图的 `final` 显示全局结果。
- 日志额外记录 `exposure_grid_active`、`exposure_grid_lr` 和
  `exposure_grid_tv_loss`；配置快照包含全部组合模式字段。

建议与独立 PPISP、独立 grid 使用相同数据划分、初始化、SH、增密策略和步数比较。
主要检查原始渲染和 PLY 的跨视角一致性；训练补偿后的指标提高本身不能证明几何改善。
