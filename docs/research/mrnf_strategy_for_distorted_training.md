# MRNF 对 3DGEER / 3DGUT 鱼眼训练的适配性调研

调研日期：2026-08-31

上游核对版本：LichtFeld Studio
[`de972c89ec6bcd27406f892b966f180a7054f2cc`](https://github.com/MrNeRF/LichtFeld-Studio/tree/de972c89ec6bcd27406f892b966f180a7054f2cc)

## 结论摘要

**MRNF 值得移植，而且它的核心 densification 信号在架构上比当前
`DefaultStrategy` 更适合 3DGEER / 3DGUT 的鱼眼畸变训练；但不能原样移植整个
`mrnf.cpp`。**

这里必须区分两层结论：

1. **有源码依据的架构结论**：MRNF 把每个像素的重建误差，按该像素上每个
   Gaussian 的实际 alpha-compositing 贡献归因回 Gaussian。在 GUT 的 world-space
   backward 中，这个归因沿相机模型和畸变参数生成的真实 ray 完成，不依赖
   `means2d.grad`，也不需要把 3D mean gradient 用针孔 Jacobian 近似还原到图像
   平面。因此它正好避开了本仓库 default 策略在畸变路径上的薄弱点。
2. **目前没有证据支持的实证结论**：LichtFeld Studio 没有公开
   “MRNF + 鱼眼”专项 benchmark 或 ablation。上游 PR 只称结果与 IGS+ 很接近，
   [自带评测脚本](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/eval/README.md)
   也只覆盖 Mip-NeRF 360。因此现在不能声称 MRNF 已被证明在鱼眼数据上优于
   default 或 MCMC；这需要在本项目中做对照实验。

推荐移植一个 **distortion-aware MRNF core**：先实现像素误差归因、候选累计、
有预算的加点/分裂和上限控制；首版关闭 `edge_map` 与
`background_improvements`。前者的上游 edge rasterizer 仍是针孔投影，后者的
far-field seed 仍按针孔模型反投影像素，直接照搬会重新引入鱼眼边缘偏差。

## MRNF 是什么

MRNF 不是当前能找到配套论文、公式推导和标准 benchmark 的独立论文方法，而是
LichtFeld Studio 内部的训练/增密策略。它最初以 `LFS densification` 的名字在
[PR #1031](https://github.com/MrNeRF/LichtFeld-Studio/pull/1031) 合入；PR 明确列出的
组成包括 growth/split、decay、noise injection、edge-guided sampling，以及从
rasterizer backward 向策略传递 error map / densification statistics。PR 作者只写了
“metrics are very comparable with IGS+”，没有给鱼眼实验。

当前实现已经远超一个可直接复制的 Python strategy：

- `mrnf.cpp` 本身约 3,600 行；
- 绑定了 LichtFeld 自有 Tensor、optimizer、free-slot/capacity 管理、序列化、
  SH 量化、裁剪/冻结和 VRAM 统计；
- 同时服务于 FastGS 与 gsplat/GUT 两条 rasterizer 路径；
- 最近还加入了 screen-share 和可选 far-field/background 机制。

所以应当移植**行为和信号定义**，而不是逐行翻译 C++ 结构。

## 核心算法

### 1. 从像素误差得到每个 Gaussian 的分数

trainer 先从 SSIM、SSIM-CS 或 RGB L1 构造二维 error map，对 mask 做处理，并在
MRNF 下按 error map 的均值归一化。相关实现见
[`trainer.cpp`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/trainer.cpp#L7062-L7166)。

对像素 `p` 和在该像素参与合成的 Gaussian `g`，world-space backward 使用与颜色
合成相同的贡献权重：

```text
w_pg = alpha_pg * T_pg
visibility_g += w_pg
error_g      += w_pg * error_map[p]
```

代码在计算 `fac = alpha * T` 后，把 `fac` 和 `fac * pixel_error` 分别原子累加到
`densification_info` 的两行。见
[`RasterizeToPixelsFromWorld3DGSBwd.cu`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/rasterization/gsplat/RasterizeToPixelsFromWorld3DGSBwd.cu#L717-L801)。

策略随后累积 visibility，并保存窗口内的最大 error score。上游虽然在 MRNF
preset 中设置了 `growth_ratio_rank=true`，但该功能受
`background_improvements` 开关控制，而 baseline profile 默认关闭后者，因此实际
仍按 raw attributed error 排序；本移植保留 visibility-ratio 作为显式可选项。见
[`MRNF::post_backward`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/strategies/mrnf.cpp#L1114-L1191)
和
[`fold_densification_and_zero_kernel`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/kernels/mrnf_kernels.cu#L203-L246)。

注意：参数名 `growth_grad_threshold` 是历史命名。当前候选条件实际比较的是
error-attribution score，不是 `means2d` gradient；移植时宜改名为
`grow_error_threshold`，避免误解。

### 2. 有预算地 grow / split

MRNF 不会像经典 ADC/default 那样把所有过阈值对象都直接 duplicate/split。
它先筛出 score 过阈值且可见的候选，再令本轮目标增长数约为：

```text
round(candidate_count * grow_fraction)
```

默认 `grow_fraction=0.07`，随后按 error score 加权、通过 Gumbel top-k 无放回
采样，且受到 population cap、fill pacing、replacement 和 oversize-split budget
约束。被裁剪数量对应的 replacement 父节点单独按可见 Gaussian 的 opacity
采样；replacement 与净增长父节点最后统一执行 IGS+ 的确定性长轴分裂，不存在
经典 ADC 的 small-Gaussian duplicate 分支。候选条件见
[`compute_refine_candidates`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/strategies/mrnf.cpp#L2343-L2348)，
预算计算见
[`grow_and_split`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/strategies/mrnf.cpp#L1958-L1990)。

这部分与鱼眼没有直接数学关系，但它能抑制“畸变边缘大量对象同时越过阈值”时
Gaussian 数量瞬时爆炸，也更容易给 3DGEER / 3DGUT 设定固定显存预算。

### 3. decay、noise 与 pruning

MRNF 还包含：

- 对低 opacity 且可见的 Gaussian 注入与 mean learning rate 相关的探索噪声；
- 连续 opacity / scale decay，而不是周期性全局 opacity reset；
- 低 opacity、退化 rotation、极小 scale、极端 scene bounds pruning；
- screen-space share cap 与 oversize split。

这些机制可能改善收敛与浮游点，但并不是 MRNF 对鱼眼更合适的核心原因，也会
显著扩大首版移植和调参范围。建议在 error attribution 跑通后分批加入。实现见
[`inject_noise` 与 `apply_decay`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/strategies/mrnf.cpp#L2585-L2632)。

当前 `mrnf_defaults()` 使用 `refine_every=200`、`stop_refine=28500`、
`max_cap=5,000,000`，但上游 Mip-NeRF 360 的 eval 配置把 cap 改成了
`1,000,000`。这说明连上游自己也没有把 preset 数值当作所有场景通用配置；移植
不应照抄 5M 上限。见
[`parameters.cpp`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/core/parameters.cpp#L657-L683)
和
[`mrnf_optimization_params.json`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/eval/mrnf_optimization_params.json)。

## 为什么核心信号更适合鱼眼

3DGUT 用 Unscented Transform 近似 Gaussian 在任意非线性投影函数下的投影，
并用 world-space Gaussian response 支持畸变相机。论文的目标本来就是在保留
rasterization 效率的同时支持 distorted cameras；其 sigma points 可经过非线性
投影函数直接投影。见
[CVPR 2025 论文与元数据](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_3DGUT_Enabling_Distorted_Cameras_and_Secondary_Rays_in_Gaussian_Splatting_CVPR_2025_paper.html)
和[官方实现](https://github.com/nv-tlabs/3dgrut)。

本仓库当前的 `DefaultStrategy` 主要依赖 `info["means2d"].grad` 的图像平面
gradient 来决定 grow/split。但 UT / GEER projection 明确返回不可微的 metadata；
当 `means2d.grad` 不存在时，本项目新增了从 `params["means"].grad` 反推图像平面
gradient 的 fallback。该 fallback 自己列出了三个限制：只支持 non-packed、假定
depth 是 camera-space Z，并且**即使相机有畸变仍使用针孔近似**。见
[`DefaultStrategy._fallback_image_plane_grads_from_means_grad`](../../gsplat/strategy/default.py#L151-L212)。

这会造成一个系统性不匹配：渲染和 loss 是沿鱼眼模型的非线性 ray 产生的，
densification 却用

```text
x = fx * X / Z + cx
y = fy * Y / Z + cy
```

的针孔 Jacobian 近似。视场中心偏差可能较小，越靠近鱼眼边缘通常越不可靠。
本项目的 3DGUT 文档也仍写着只支持 MCMC densification；当前 default fallback
可以视为后来加入的兼容路径，但它并没有消除上述近似。见
[`docs/3dgut.md`](../3dgut.md#L13-L19)。

相比之下，上游 MRNF 的 GUT backward 在构造 pixel ray 时接收
`camera_model_type`、intrinsics、radial/tangential/thin-prism coefficients，再把该
ray 上实际参与 alpha compositing 的 Gaussian 与 pixel error 关联。见
[`init_pixel_bwd`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/rasterization/gsplat/RasterizeToPixelsFromWorld3DGSBwd.cu#L42-L100)。

本仓库已经具备同样合适的接入点：

- Eval3D backward 已接收 camera model 与各类 distortion coefficients：
  [`_RasterizeToPixelsEval3D`](../../gsplat/cuda/_wrapper.py#L1655-L1930)；
- CUDA backward 已通过 `OpenCVFisheyeCameraModel` 从像素构造真实 world ray：
  [`RasterizeToPixelsFromWorld3DGSBwd.cu`](../../gsplat/cuda/csrc/RasterizeToPixelsFromWorld3DGSBwd.cu#L151-L239)；
- 同一 kernel 已计算 `fac = alpha * T`，正好可以在此增加 visibility/error 两个
  accumulator：
  [`fac` 计算位置](../../gsplat/cuda/csrc/RasterizeToPixelsFromWorld3DGSBwd.cu#L367-L387)。

因此，MRNF core 不是“给鱼眼打一个额外权重”，而是让 densification 与已经正确
处理畸变的渲染路径共享同一条 pixel-to-ray-to-Gaussian attribution。这是它相对
当前 default fallback 最实质的优势。

## 不能原样移植的两处针孔假设

### 1. 默认启用的 edge guidance

上游 MRNF 默认 `use_edge_map=true`。它先对 target image 做 Canny，再用单独的
edge rasterizer 把边缘分数归因到 Gaussian，最终将 normalized edge score 以
`1 + 0.25 * score` 乘到 growth weights 上。

但这个 edge rasterizer 只向 CUDA API 传 `w2c`、`fx/fy/cx/cy`、图像尺寸和
Gaussian 参数，没有传 camera model、radial/tangential/thin-prism coefficients，
所以它是针孔投影接口。见
[`edge_rasterizer.cpp`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/rasterization/edge_rasterizer.cpp#L13-L72)。

这意味着“MRNF core 的 GUT error attribution 是畸变感知的”和“MRNF 默认的所有
辅助信号都是畸变感知的”不能混为一谈。首版应设置 `use_edge_map=false`。如果
后续要恢复 edge guidance，应复用 Eval3D 的真实 ray attribution，而不是移植这个
edge rasterizer。

### 2. 可选 background improvements / far seeding

`background_improvements` 在当前 MRNF preset 中默认关闭，因此不妨碍首版移植。
但如果以后启用，需要先修正两处：

- `accumulate_explore_sample` 在 render metadata 不可用时，只用
  `fx/fy/cx/cy` 投影 Gaussian center；
- `seed_from_view` 用 `u=(x-cx)/fx`、`v=(y-cy)/fy` 和
  `(u*t, v*t, t)` 反投影采样像素。

后者的针孔反投影可直接见
[`seed_from_view`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/strategies/mrnf.cpp#L3139-L3174)。
鱼眼版本必须改用本仓库已有的 `compute_raymap` / camera-model unprojection。

## 与现有策略的定位

| 策略 | densification 主要依据 | 对畸变投影的关系 | 适合作为 |
| --- | --- | --- | --- |
| 当前 Default | `means2d.grad`；Eval3D 下退化为针孔式 3D-gradient fallback | 在 UT/GEER 下存在明确近似 | 现有稳定基线 |
| 当前 MCMC | opacity 分布采样、低 opacity relocation、position noise | 不依赖 2D projection gradient，因此不会被针孔 Jacobian 直接破坏；但不按像素误差定向修补 | renderer-agnostic 基线 |
| MRNF core | 实际 compositing contribution 加权的 pixel error | 放在 Eval3D world-ray backward 中时天然跟随鱼眼 ray | 最值得新增的 error-driven 策略 |

MCMC 的优点是简单、与相机投影解耦；缺点是新 Gaussian 主要按 opacity 分布采样，
不能像 MRNF 那样直接把“哪一块鱼眼图像仍有高残差”归因给负责的 Gaussian。
所以建议保留 MCMC 作为对照，而不是由 MRNF 直接替代。

## 建议的移植范围

### Phase 1：distortion-aware MRNF core

这是最小但闭环的版本。

1. 新增 `gsplat/strategy/mrnf.py`，复用已有 strategy ops 与 optimizer-state
   mutation，不移植 LichtFeld 的自定义 Tensor/free-slot/serialization 层。
2. state 先只保留：
   - `visibility_sum[N]`；
   - `error_max[N]`；
   - 可选 `error_ratio_max[N]`。
3. 给 `_RasterizeToPixelsEval3D` 增加 optional `densification_error_map` 和
   `[2, N]` `densification_info` buffer；在现有 world-space CUDA backward 的
   `fac = alpha * T` 位置累加：

   ```text
   densification_info[0, g] += fac
   densification_info[1, g] += fac * error_map[p]
   ```

4. trainer 先用 detached mean absolute RGB error；对 fisheye invalid mask 置零，
   再按**有效像素均值**归一化。稳定后再切换/增加 SSIM-CS error map。
5. 每个 refine window：
   - 用 raw error threshold 选候选；
   - baseline 用 raw error 排序或采样，并把 `error / visibility^p` 保留为可选项；
   - 目标数为候选数乘 `grow_fraction`；
   - 用 `torch.multinomial(replacement=False)` 或 Gumbel top-k；
   - replacement 按可见 Gaussian 的 opacity 独立采样；
   - 所有选中对象都执行确定性 long-axis split，并执行 opacity pruning；
   - 应用 `max_gaussians` 上限。
6. 首版固定：
   - `use_edge_map=false`；
   - `background_improvements=false`；
   - 不加入 screen-share、far seed、free-slot allocator、SH quantization、Vulkan
     和 LichtFeld serialization。

Phase 1 的关键验收不是“代码里出现 `MRNFStrategy`”，而是确认
`densification_info[1]` 会随高误差像素及其实际贡献 Gaussian 改变，并且整个信号
不经过针孔 Jacobian。

### Phase 2：MRNF 优化机制

在 core 证明有效后，再分别消融加入：

- opacity / scale decay；
- low-opacity exploration noise；
- visibility-ratio rank；
- screen-share cap 与 oversize split；
- 更精细的 SSIM-CS error map。

每次只增加一类机制，避免把“误差归因有效”与“上游超参数组合有效”混在一起。

### Phase 3：真正 distortion-aware 的辅助信号

- edge guidance：用同一套真实 ray/compositing contribution 对 Canny map 做归因；
- far-field seed：用 `compute_raymap` 或相机模型的 image-to-camera-ray 反投影；
- 如果训练目标希望按 solid angle 而不是按 image pixel 等权，可额外研究鱼眼
  pixel solid-angle/Jacobian weighting。这不是原版 MRNF 的功能，也不应在首版
  默认启用。

## 推荐验证设计

当前证据足以支持移植原型，但不足以宣称最终质量更高。建议在同一数据、初始化、
随机种子、训练步数和 Gaussian cap 下做：

1. Default（当前 pinhole-style fallback）；
2. MCMC；
3. MRNF core，关闭 edge/background；
4. MRNF core + 后续实现的 distortion-aware edge guidance。

除全图 PSNR / SSIM / LPIPS 外，还应报告：

- 按归一化图像半径分桶的 PSNR/SSIM，尤其是最外层鱼眼区域；
- Gaussian 数量曲线、峰值显存和每轮新增数量；
- 无效 fisheye 像素是否完全被 error-map normalization 排除；
- 边缘/高频区域 residual 与 floaters；
- 相同 cap 下的收敛速度。

全图平均指标可能掩盖视场边缘的改善，radial-bin 指标才直接检验这次移植的核心
假设。

## 风险与实现注意事项

- **阈值不能直接照搬**：上游 error map 归一化、图像 layout、alpha activation、
  batch size 与本仓库不同；`0.003` 只能作参考。
- **mask normalization**：若先把无效鱼眼像素置零再对整张图求均值，大面积无效
  区会放大有效区 score。应以 valid-pixel mean 为分母。
- **多相机/packed 索引**：上游当前 kernel 对 dense/packed 与多 camera 有自己的
  索引约定；本仓库首版可先限定 simple trainer 的 non-packed batch=1，再明确
  扩展。
- **副作用 buffer 生命周期**：`densification_info` 会在 autograd backward 内原地
  累加，strategy refine 改变 N 后必须重建/resize 并清零。
- **误差与曝光模型**：PPISP、bilateral grid 或可学习曝光开启时，error map 应取
  最终参与 photometric loss 的颜色空间，同时保持 detached，避免引入第二条梯度
  路径。
- **上游仍在快速变化**：移植与引用应固定上述 commit，不跟随 `master` 不加审查
  地同步。

## 许可证

LichtFeld Studio 是 GPL-3.0-or-later，本仓库是 AGPL-3.0。FSF 明确说明 GPLv3 与
AGPLv3 可以组成同一 combined program，组合整体适用 AGPL；但不能简单把原 GPL
代码重新声明为 AGPL。若直接改写/复制上游实现，应保留版权和 SPDX 来源说明，并
在提交中注明固定 commit。见
[GNU License Compatibility](https://www.gnu.org/licenses/license-compatibility.en.html)
和
[LichtFeld Studio LICENSE](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/LICENSE)。

## 最终判断

可以把结论压缩成一句话：

> **MRNF 不是一个已经通过鱼眼 benchmark 证明更好的策略，但它的核心像素误差
> 归因机制与 3DGEER / 3DGUT 的真实畸变 ray 路径高度匹配，明显比当前 default 的
> 针孔式 fallback 更合理；最值得移植的是 MRNF core，而不是默认开启的全部
> LichtFeld 特性。**

因此建议进入 Phase 1 实现，并把 Default 与 MCMC 都保留为基线。只有完成同 cap、
同 seed、包含 radial-bin 指标的对照后，才能把“更合理”升级为“在本项目鱼眼训练
上更好”。
