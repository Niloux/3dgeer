# 3DGEER 与 LichtFeld Studio 深度约束对比

调研日期：2026-09-01

LichtFeld Studio 固定版本：[`bc8189a4c596f0f97893669562410f12b31628eb`](https://github.com/MrNeRF/LichtFeld-Studio/tree/bc8189a4c596f0f97893669562410f12b31628eb)

## 结论

对当前项目的 **LiDAR、米制、camera-Z、原始鱼眼像素域** 深度图，现有 3DGEER
方案更合适：它不重新拟合可靠的绝对尺度，且监督直接走当前 distortion-aware
3DGUT/3DGEER 渲染与相机优化图。LichtFeld Studio（LFS）的深度 loss 只在
`gut=false` 的 FastGS backward 上执行，anchor 又是纯针孔投影，不能原样替代当前
方案。[LFS trainer 路径](https://github.com/MrNeRF/LichtFeld-Studio/blob/bc8189a4c596f0f97893669562410f12b31628eb/src/training/trainer.cpp#L5981-L6041)
[LFS depth-loss gate](https://github.com/MrNeRF/LichtFeld-Studio/blob/bc8189a4c596f0f97893669562410f12b31628eb/src/training/trainer.cpp#L6814-L6820)
[LFS anchor 投影](https://github.com/MrNeRF/LichtFeld-Studio/blob/bc8189a4c596f0f97893669562410f12b31628eb/src/training/kernels/depth_loss.cu#L444-L500)
当前 master 的实际 guard 链是
`fastgs_path = !optimization.gut` →
`run_fastgs_gaussian_backward = fastgs_path && update_gaussians_this_iter` →
depth-loss block 要求 `run_fastgs_gaussian_backward`。因此 LFS 自己的 GUT distorted
path 不会运行这项 loss；本项目 `keep_distortion + GEER/eval3d` 路径只能移植其数学
思路，不能复用上游调用路径。

但若输入是 **单目网络产生的相对深度/视差，或含明显离群点与量化台阶的稠密
prior**，LFS 的 loss 设计明显更强：固定的点云定标、Geman–McClure、邻域梯度项、
alpha/ROI 置信权重、量化 deadband 和随训练衰减的权重，都比当前单一 inverse-depth
L1 更稳健。[LFS loss 接口与设计说明](https://github.com/MrNeRF/LichtFeld-Studio/blob/bc8189a4c596f0f97893669562410f12b31628eb/src/training/kernels/depth_loss.hpp#L115-L141)

两者都不是“把某个 Gaussian 拉到 GT 表面”：二者约束的都是同一个一阶统计量——
射线上参与合成的 Gaussian 中心 camera-Z 的 alpha-compositing 期望。因此“一近一远
但期望值正确”的多层解在两边都仍然可能存在。

## 精确比较

令 `w_i = T_i alpha_i`、`A = sum_i w_i`、`R = sum_i w_i z_i`，其中 `z_i` 是
Gaussian 中心的 camera-space Z。LFS FastGS 明确计算 `z_i` 并累加
`R`；3DGEER 的 `RGB+ED` 将同样的累计深度除以累计 alpha。
[LFS Gaussian Z](https://github.com/MrNeRF/LichtFeld-Studio/blob/bc8189a4c596f0f97893669562410f12b31628eb/src/training/rasterization/fastgs/rasterization/include/kernels_forward.cuh#L117-L124)
[LFS 合成](https://github.com/MrNeRF/LichtFeld-Studio/blob/bc8189a4c596f0f97893669562410f12b31628eb/src/training/rasterization/fastgs/rasterization/include/kernels_forward.cuh#L772-L807)
[3DGEER ED](../../gsplat/rendering.py#L1218-L1226)

| 项目 | 当前 3DGEER | LichtFeld Studio |
| --- | --- | --- |
| 预测统计量 | `e = R / max(A, 1e-10)` | loss 内部 `e = max(R, 0) / A` |
| target | 已解码并变换到训练世界单位的米制 camera-Z | 任意正值相对 prior，经每相机固定 affine anchor 变成 Z 或 softened inverse-Z |
| 主残差 | `abs(1/e - 1/z_gt)` | 标准化、deadband 后的 softened inverse-depth Geman–McClure |
| 空间项 | 无 | 右/下相邻像素 inverse-depth 差的同类 robust loss |
| 像素权重 | 二值 `target>0`，再与数据/相机 mask 相交 | `A * ROI_weight`；要求 `A>1e-3`，无 ROI 时为 `A` |
| 全局权重 | 默认 `0.01`，训练全程恒定；再乘 `scene_scale` | 默认初值 `2.0`，指数衰减到 `0.04` |
| 默认开关 | dataclass 默认关闭；当前 YAML profile 开启 | 默认关闭，mode 默认 `ssi` |
| 相机/渲染支持 | 当前 fisheye + GEER/eval3d 路径 | 当前 depth loss 仅 FastGS，`gut=true` 不执行 |

3DGEER 对稠密 sidecar 的实际 loss 为

```text
M = (z_gt > 0) & optional_mask
L_ours = depth_lambda * scene_scale
         * sum(M * abs(where(e > 0, 1/e, 0) - 1/z_gt)) / max(sum(M), 1)
```

实现见 [训练 loss](../../examples/simple_trainer.py#L1875-L1920)。sidecar 必须是
单通道 uint16，`0` 为 invalid；有效值按 `(q-1)/65534 * depth_max *
depth_world_scale` 解码。稀疏深度降采样时保留落入训练像素的最近 Z，而不是插值
invalid zero。[数据解码](../../examples/datasets/colmap.py#L679-L705)
[稀疏降采样](../../examples/datasets/colmap.py#L87-L128)

这一点也不同于 LFS：其 target 与 rendered depth 尺寸不一致时直接做 Lanczos
grayscale resize。Lanczos 适合它通常使用的稠密单目 prior；对带大量 invalid zero
的稀疏 LiDAR z-buffer 会混合无效值，不能照搬。
[LFS target resize](https://github.com/MrNeRF/LichtFeld-Studio/blob/bc8189a4c596f0f97893669562410f12b31628eb/src/training/trainer.cpp#L6862-L6869)

LFS 先用初始点云为每个相机拟合一次固定 anchor。设 raw prior 为 `t`，
`f = 0.05 * median(z_anchor)`，它同时尝试

```text
disparity model: 1 / (z_anchor + f) = a*t + b
depth model:                     z_anchor = a*t + b
```

拟合至少需要 256 个样本、30% inlier 和 `|corr| >= 0.35`；`ssi` 在数据集级选择
累计相关性更好的 prior 类型，并丢弃类型不可用或 slope 符号不一致的相机。
[anchor 拟合阈值与公式](https://github.com/MrNeRF/LichtFeld-Studio/blob/bc8189a4c596f0f97893669562410f12b31628eb/src/training/kernels/depth_loss.cu#L435-L440)
[anchor 拟合实现](https://github.com/MrNeRF/LichtFeld-Studio/blob/bc8189a4c596f0f97893669562410f12b31628eb/src/training/kernels/depth_loss.cu#L744-L823)
[数据集级选择](https://github.com/MrNeRF/LichtFeld-Studio/blob/bc8189a4c596f0f97893669562410f12b31628eb/src/training/trainer.cpp#L2498-L2557)

训练时，LFS 令 `p = 1/(e+f)`，并把 target 转成

```text
disparity model: d = min(a*t+b, 1/f)
depth model:     d = 1/(a*t+b+f)
```

有效像素要求 `t>0`、`A>1e-3`，且 `t`、`R`、`A` 都有限，外加可选 ROI weight
为有限正值；
没有独立的外部 confidence map 接口。普通 photometric segmentation mask 不会传给
depth kernel；只有 depth 中的 zero-invalid 和可选 cropbox ROI 生效。启用 cropbox 时
内部权重为 `1`、外部默认 `0.1`。
[ROI map](https://github.com/MrNeRF/LichtFeld-Studio/blob/bc8189a4c596f0f97893669562410f12b31628eb/src/training/kernels/roi_weight_map.cu#L99-L144)
[ROI 默认值](https://github.com/MrNeRF/LichtFeld-Studio/blob/bc8189a4c596f0f97893669562410f12b31628eb/src/core/include/core/parameters.hpp#L159-L162)
令 `omega_i=A_i*roi_i`、
`s=2*sigma_omega(p)`、`rho(x)=0.5*x^2/(1+x^2)`、
`db(r,delta)=sign(r)max(|r|-delta,0)`，则其标量 loss 为

```text
L_lfs = lambda(k) / sum_i(omega_i) * [
  sum_i omega_i * rho(db(p_i-d_i, delta_i) / s)
  + sum_(i,j in right/down edges) min(omega_i,omega_j)
      * rho(db((p_j-p_i)-(d_j-d_i), delta_i+delta_j) / s)
]
```

上式假定 affine 后的 target 有效。严格对应 kernel 时，`sum(omega)` 与 `sigma_p`
先在 anchor 变换前的 active pixels 上统计；若随后 `a*t+b<=0`，该像素不产生残差，
但仍留在这两个 detached statistics 中。

其中梯度项内部权重固定为 `1`；若 prior 量化步长为 `q`，先取
`h=0.5*q*|a|`，disparity model 用 `delta=h`，depth model 用
`delta=h*d^2`。[有效像素与 robust 函数](https://github.com/MrNeRF/LichtFeld-Studio/blob/bc8189a4c596f0f97893669562410f12b31628eb/src/training/kernels/depth_loss.cu#L35-L75)
[预测/target 变换](https://github.com/MrNeRF/LichtFeld-Studio/blob/bc8189a4c596f0f97893669562410f12b31628eb/src/training/kernels/depth_loss.cu#L251-L305)
[data 与梯度项](https://github.com/MrNeRF/LichtFeld-Studio/blob/bc8189a4c596f0f97893669562410f12b31628eb/src/training/kernels/depth_loss.cu#L309-L432)
[loss 汇总](https://github.com/MrNeRF/LichtFeld-Studio/blob/bc8189a4c596f0f97893669562410f12b31628eb/src/training/kernels/depth_loss.cu#L631-L647)

LFS 的 hard-coded schedule 是

```text
lambda(k) = depth_loss_weight * 0.02 ^ min(k / total_iterations, 1)
```

默认 `use_depth_loss=false / depth_loss_weight=2.0 / depth_loss_mode=ssi`。
[默认值](https://github.com/MrNeRF/LichtFeld-Studio/blob/bc8189a4c596f0f97893669562410f12b31628eb/src/core/include/core/parameters.hpp#L196-L200)
[schedule 常量](https://github.com/MrNeRF/LichtFeld-Studio/blob/bc8189a4c596f0f97893669562410f12b31628eb/src/training/trainer.cpp#L424-L425)
[schedule 应用](https://github.com/MrNeRF/LichtFeld-Studio/blob/bc8189a4c596f0f97893669562410f12b31628eb/src/training/trainer.cpp#L6881-L6933)

LFS 没有“anchor 失败后仍做相对 SSI”的 fallback：没有初始点云时不拟合，某相机
没有有效 anchor 时该帧的 depth loss 不执行。
[anchor 硬门槛](https://github.com/MrNeRF/LichtFeld-Studio/blob/bc8189a4c596f0f97893669562410f12b31628eb/src/training/trainer.cpp#L2424-L2447)
[逐帧执行条件](https://github.com/MrNeRF/LichtFeld-Studio/blob/bc8189a4c596f0f97893669562410f12b31628eb/src/training/trainer.cpp#L6903-L6934)

## 梯度到哪里

3DGEER 直接对 `RGB+ED` 的 PyTorch 图反传。深度项会更新 Gaussian `means`，也会
通过合成权重更新 `opacities`、`scales`、`quats`；它不依赖 SH/color 或后处理颜色
模块。当前 profile 还启用了 pose 与 calibration optimization，所以在各自 schedule
激活后，深度项也能沿传入 renderer 的 `camtoworlds`、`K` 和 fisheye radial
coefficients 更新相机参数。[renderer 参数流](../../examples/simple_trainer.py#L1487-L1540)
[训练相机参数流](../../examples/simple_trainer.py#L1741-L1813)
[当前 profile](../../configs/simple_trainer/simple_trainer.yaml#L16-L32)

LFS 手写梯度同时输出 `dL/dR` 与 `dL/dA`，因此可移动 means，也会经 alpha footprint
影响 opacity/scale/rotation；anchor 与 prior 固定，depth 项不更新颜色。
[显式 `R/A` 梯度](https://github.com/MrNeRF/LichtFeld-Studio/blob/bc8189a4c596f0f97893669562410f12b31628eb/src/training/kernels/depth_loss.cu#L422-L430)
[FastGS blend backward](https://github.com/MrNeRF/LichtFeld-Studio/blob/bc8189a4c596f0f97893669562410f12b31628eb/src/training/rasterization/fastgs/rasterization/include/kernels_backward.cuh#L918-L944)
严格说，LFS 的显式梯度把 `omega`、归一化分母和 `sigma_p` 当作 detached statistics；
`dL/dA` 来自 `e=R/A`，不是对 alpha 置信权本身求完整导数。虽然 FastGS kernel具备
可选 `grad_w2c`，trainer 当前传 `nullptr`，所以这项 loss 不优化相机外参。
[FastGS backward 调用](https://github.com/MrNeRF/LichtFeld-Studio/blob/bc8189a4c596f0f97893669562410f12b31628eb/src/training/rasterization/fast_rasterizer.cpp#L812-L849)

## 取舍与建议

- **继续使用当前方案作为基线。** 对可信 metric LiDAR，LFS 的 per-camera affine
  anchor 没有必要，还可能把本来统一的绝对尺度变成逐相机拟合尺度。
- **值得移植 LFS 的 robust 外壳，而不是整套 anchor。** 优先级建议为：有效性与
  finite/alpha 门槛、置信权重、随训练衰减；数据仍有离群点时再加 Geman–McClure；
  depth 足够稠密连续时再评估邻域梯度项和量化 deadband。
- **不要期待任一方案单独解决射线多层歧义。** 若目标是让单个 Gaussian 真正贴住
  LiDAR 表面，还需要中值/首表面深度、ray-termination 分布约束，或直接的
  per-Gaussian/point-to-surface 几何项；这与当前两个 expected-depth loss 是另一类约束。
