# PPISP 调研与 3dgeer 接入建议

调研日期：2026-08-26

## 结论摘要

这里的 PPISP 指 NVIDIA Spatial Intelligence Lab 的 **PPISP: Physically-Plausible Compensation and Control of Photometric Variations in Radiance Field Reconstruction**，为 CVPR 2026 Oral 工作。其目标是在辐射场渲染之后加入一个可微、低容量且物理含义明确的相机成像模型，避免相机曝光、白平衡、渐晕和响应曲线差异被场景颜色或几何吸收。[项目页](https://research.nvidia.com/labs/sil/projects/ppisp/)，[论文 v2](https://arxiv.org/html/2601.18336)，[官方代码](https://github.com/nv-tlabs/ppisp)

对本项目而言，PPISP **值得作为 bilateral grid 的替代方案做一组独立实验**，尤其适合处理：

- 每帧自动曝光变化；
- 每帧自动白平衡或全局色偏；
- 每个相机固定的径向渐晕；
- 每个相机固定的非线性 CRF/tone curve。

但它不是一个更强的局部校正器。论文明确说明，PPISP 有意限制容量，不能拟合局部 tone mapping、lens flare 和类似空间变化。因此它不会直接解决随视角移动的路面镜面高光、局部移动阴影或任意局部 ISP。[论文 5.4 与 Limitations](https://arxiv.org/html/2601.18336#S5.SS4)

推荐的首轮方案是：

1. `PPISP-only`，关闭现有 bilateral grid；
2. 暂时关闭 PPISP controller，只验证它是否能得到更稳定的 canonical SH0 PLY；
3. PLY 继续导出 PPISP 之前的 raw/canonical Gaussian 颜色；
4. checkpoint 单独保存 PPISP 状态，并同时输出 canonical 与 PPISP 两套训练视图；
5. 与现有 `full-bilateral_grid15000` 做严格同配置对照，不以训练 PSNR 作为唯一指标。

## 方法结构

PPISP 在场景渲染得到的 radiance 后依次应用四个模块：[论文 Method](https://arxiv.org/html/2601.18336#S4)

```text
Gaussian/NeRF raw radiance
    -> per-frame exposure
    -> per-camera chromatic vignetting
    -> per-frame color correction
    -> per-camera CRF
    -> supervision image
```

### 1. Exposure offset

每帧学习一个全局曝光标量：

```text
I_exp = L * 2^delta_t
```

它只允许改变整张图的亮度，意在覆盖快门、光圈、模拟/数字增益和自动曝光变化。[论文 4.1](https://arxiv.org/html/2601.18336#S4.SS1)

### 2. Vignetting

每个相机、每个 RGB 通道学习一个径向衰减模型，其参数包括光心以及 `r^2/r^4/r^6` 三项多项式系数。这个结构能解释镜头固定渐晕，但不能表达任意形状的局部明暗变化。[论文 4.2](https://arxiv.org/html/2601.18336#S4.SS2)

### 3. Color correction

每帧学习 8 个色度控制参数，通过 RG chromaticity homography 做全局颜色校正，并额外归一化强度，使白平衡/色偏尽量与曝光解耦。[论文 4.3](https://arxiv.org/html/2601.18336#S4.SS3)

### 4. Camera response function

每个相机、每个通道学习 4 参数的单调 S 型分段幂函数，再组合 gamma。它建模相机固定的非线性响应和全局 tone curve，且通过结构保证平滑、单调。[论文 4.4](https://arxiv.org/html/2601.18336#S4.SS4)

官方实现中的主要参数形状为：

| 参数 | 粒度 | 形状 |
| --- | --- | --- |
| exposure | 每帧 | `[num_frames]` |
| color correction | 每帧 | `[num_frames, 8]` |
| vignetting | 每相机、每通道 | `[num_cameras, 3, 5]` |
| CRF | 每相机、每通道 | `[num_cameras, 3, 4]` |

这些定义可在[官方实现](https://raw.githubusercontent.com/nv-tlabs/ppisp/main/ppisp/__init__.py)中直接核对。

## 训练与推理

论文实验先联合训练 scene representation 与 PPISP 30k iterations，然后冻结 scene 和 PPISP 参数，再训练 controller 5k iterations。[论文实验设置](https://arxiv.org/html/2601.18336#S5)

Controller 是每相机一个网络。它读取 novel view 的 raw rendered radiance，通过 `1x1` 卷积、池化到 `5x5` 网格和 MLP，预测该视角的全局 exposure 与 color correction，作用类似真实相机的 AE/AWB。每相机固定的 vignetting 与 CRF 则直接复用。[论文 4.5](https://arxiv.org/html/2601.18336#S4.SS5)

当前官方包默认采用 distillation：在总训练步数约 80% 时开始 controller 阶段；PPISP 参数被冻结，输入 radiance 被 detach，调用方还应同时冻结 Gaussians/NeRF。[官方集成说明](https://github.com/nv-tlabs/ppisp#controller-distillation-mode)

如果当前目标只是验证 canonical SH0 PLY，controller 首轮可以关闭，原因是：

- controller 只决定 novel-view 的显示曝光和白平衡，不进入 PLY；
- 开启默认 distillation 会让场景在最后约 20% 训练阶段停止更新；
- 先关闭 controller 更容易判断 canonical 场景本身是否改善。

后续如果 viewer 也要输出“像相机拍摄出来的”novel view，再增加 controller 阶段。

## 对局部高光、阴影和 ISP 的能力边界

| 变化来源 | PPISP 能力 | 原因 |
| --- | --- | --- |
| 每帧整体曝光 | 强 | 显式的 per-frame exposure |
| 每帧整体白平衡/色偏 | 强 | 显式的 per-frame chromaticity transform |
| 固定镜头渐晕 | 强 | per-camera、per-channel radial model |
| 固定相机全局 tone curve/CRF | 强 | per-camera nonlinear CRF |
| 手机局部 tone mapping | 弱/不能 | 同时依赖空间与强度，论文明确列为限制 |
| 固定在图像坐标中的复杂 ISP | 弱 | 除径向渐晕外没有任意 2D 空间场 |
| 随视角移动的镜面高光 | 不能 | 这是表面反射/观察方向效应，不是相机 ISP |
| 局部移动阴影 | 不能 | 这是动态照明，不是全局相机参数 |
| 静态阴影 | 会进入 canonical 场景 | PPISP 恢复的是带固定照明的 scene radiance，不是材质 albedo |

因此，对当前“脚下地面在不同俯视角度变化明显”的问题，需要先区分：

- 如果变化来自相机 AE/AWB、镜头渐晕或全局 CRF，PPISP 很对症；
- 如果是路面真实镜面高光、移动阴影或局部 tone mapping，PPISP-only 仍可能让残差进入 SH0、opacity 或几何；
- SH 固定为 0 并不会让最终像素与视角无关，因为投影覆盖、alpha 混合和遮挡顺序仍随视角变化。

## 与现有 bilateral grid 的差异

本项目当前 bilateral grid 为每张训练图维护一个 `[12, 8, 16, 16]` 的网格，根据像素位置和当前 RGB 灰度切片出局部 `3x4` affine color transform。它在训练 loss 前应用，并使用 TV loss 正则化。[当前实现](../examples/lib_bilagrid.py#L180)，[训练接入](../examples/simple_trainer.py#L1777)

| 维度 | 当前 bilateral grid | PPISP |
| --- | --- | --- |
| 表达能力 | 高；每图、空间与强度相关 | 低；受物理结构限制 |
| 局部 tone mapping | 能拟合一部分 | 明确不擅长 |
| 曝光/白平衡解释性 | 低 | 高 |
| 固定渐晕/CRF 的跨视角复用 | 没有显式分解 | 有 per-camera 模块 |
| 训练视图拟合 | 通常更高 | 通常略低 |
| novel-view 泛化 | 容易因逐图高容量而变差 | 设计目标就是减少过拟合 |
| novel-view 参数 | 没有天然定义 | controller 预测 AE/AWB |

论文在 Tanks & Temples 上报告：

| 方法 | Train-view PSNR | Novel-view PSNR |
| --- | ---: | ---: |
| BilaRF | 26.87 | 19.78 |
| PPISP + BilaRF | 26.66 | 23.52 |
| PPISP | 25.85 | 24.62 |

论文的解释是，bilateral grid 的高容量提高了训练视图拟合，但会记住每张训练图的局部变化；PPISP 的受限容量牺牲一些训练 PSNR，换取 novel-view 泛化。[论文 5.4 / Table 5](https://arxiv.org/html/2601.18336#S5.SS4)

这不表示 bilateral 没有价值。它更适合“确实需要吸收局部 ISP”的数据，但当前目标是稳定的、可导出的 canonical SH0 PLY，因此应先验证 PPISP-only。若后续确认仍需要局部残差，可再试：

- PPISP 先训练；
- bilateral 延迟启用；
- 使用更低分辨率 grid；
- 增强 identity/TV 正则；
- 用跨视角 canonical 稳定性而非 training PSNR 选模型。

论文自己的 hybrid 实验也采用了先延迟 bilateral 的策略，并观察到局部模块增加后 novel-view 指标下降。[论文优化设置](https://arxiv.org/html/2601.18336#S10.SS1)

## 对 canonical SH0 PLY 的影响

PPISP 是 rasterization 之后的图像处理模块，标准 3DGS PLY 无法存储 per-frame exposure/color、per-camera vignetting/CRF 或 controller。

推荐导出关系为：

```text
checkpoint = canonical Gaussians + PPISP state (+ optional controller)
PLY        = canonical Gaussians only
viewer     = render PLY -> optional PPISP post-process
```

PPISP 通过曝光均值、色度偏移均值、通道方差和渐晕物理约束来减少 scene radiance 与 ISP 之间的 gauge ambiguity。[论文 4.6](https://arxiv.org/html/2601.18336#S4.SS6) 但这并不保证 PLY 颜色是绝对物理 albedo：

- 它仍包含静态光照和静态阴影；
- radiance 与 CRF 仍存在指数歧义，论文也单独讨论了可辨识性问题；[论文 Appendix A.3](https://arxiv.org/html/2601.18336#A3)
- 丢掉 PPISP 后直接在通用 PLY viewer 中查看，亮度、对比度和输入 LDR 不完全一致是可能的；
- 尤其是 CRF 为非线性且在 alpha compositing 之后应用，不能严格等价地逐 Gaussian 烘焙到 SH0；渐晕又依赖相机像素位置，更不应烘焙到 3D 颜色。

所以，PPISP 可以让 raw PLY 的跨帧颜色 gauge 更一致，但不能保证“裸 PLY 的显示外观”和训练照片一致。如果部署端只能读取普通 PLY，应将“裸 PLY 观感”作为单独验收项；如果允许自定义 viewer，则应一并保存 PPISP state。

## 与当前仓库的接入点

当前 trainer 已经具备 PPISP 接入所需的大部分结构：

- render 后、photometric loss 前的 appearance adapter 插入点：[训练循环](../examples/simple_trainer.py#L1753)；
- 唯一训练图索引 `image_id`，可作为 PPISP 的 per-frame index：[Dataset](../examples/datasets/colmap.py#L646)；左右相机共享的 rig `frame_id` 不能直接用于这里，否则两张图会错误共享曝光和颜色参数；
- `camera_id` 可作为 per-camera 参数来源，但需要先压缩映射到 `0..num_cameras-1`；
- checkpoint 已有附加 appearance module state 的保存/恢复模式；
- 评估已经支持 canonical 与 bilateral 两套输出；
- PLY 当前明确导出 correction 之前的 canonical SH color。

推荐的最小接入方案：

1. 增加可选依赖并固定官方 release tag，建议当前最新 `v1.2.1`，不要跟随未固定的 `main`。[官方 tags](https://github.com/nv-tlabs/ppisp/tags)
2. 增加 `use_ppisp` 与少量必要配置，保证它与 `use_bilateral_grid`、`app_opt` 首轮互斥。
3. 初始化 `PPISP(num_cameras, num_frames=len(trainset))`，建立 raw COLMAP camera id 到连续 PPISP camera index 的映射。
4. 为每张训练图建立唯一的 PPISP frame index，而不是使用会被多相机共享的 rig timestamp `frame_id`。直接使用当前训练集内连续的 `image_id` 可以驱动核心模块；如果要调用官方 `frames_per_camera` 报告接口，则应改成按 camera 分组的连续索引，或自行按 camera 收集参数，否则报告会把交错排列的左右相机帧切错。
5. 在 `render_scene()` 之后、L1/SSIM 之前应用 PPISP。
6. 将 `ppisp.get_regularization_loss()` 加入总 loss，并纳入独立 optimizer/scheduler。
7. checkpoint 保存 `ppisp.state_dict()`；评估时严格恢复。
8. 训练集评估同时输出：
   - `train_step...png`：raw canonical；
   - `train_ppisp_step...png`：应用对应训练帧参数后的结果。
9. PLY 保持 raw/canonical；另外输出 PPISP 参数报告或 JSON，便于确认 exposure、vignetting、color、CRF 各自吸收了什么。
10. 第一轮 `use_controller=False`。只有当自定义 viewer/novel-view 输出也要模拟 AE/AWB 时，再增加独立 controller 阶段。

当前数据保持畸变并使用鱼眼渲染，这不妨碍 PPISP：渐晕是在最终传感器像素坐标中计算的，而且光心可学习。完整图像输入时可以让 `pixel_coords=None`，由官方 kernel 使用像素中心。如果以后启用随机 patch，则必须传入 patch 在完整图像中的原始像素坐标与完整分辨率，不能让 PPISP 把 patch 中心误当成镜头光心。鱼眼有效区域外的黑色 mask 还可能影响 controller 的测光统计，因此 controller 阶段应单独检查这一点。

## 推荐首轮实验配置

论文和官方实现的默认 PPISP 主参数优化设置是 Adam `lr=0.002`，500 步 warmup，从 `0.01x` 学习率起步并指数衰减到 `0.01x`。正则项权重大致为 brightness mean `1.0`、color mean `1.0`、channel variance `0.1`、vignetting constraint `0.01`。[论文 Appendix C.1](https://arxiv.org/html/2601.18336#S10.SS1)

本项目当前训练为 15k steps，因此 scheduler 的 decay horizon 应跟随本次总步数设为 15k，而不是机械保留论文的 30k。

建议实验矩阵：

| 实验 | bilateral | PPISP | controller | 目的 |
| --- | --- | --- | --- | --- |
| A（已有） | 16x16x8 | off | off | 高容量局部校正基线 |
| B（首轮） | off | on | off | 验证 canonical PLY 与几何稳定性 |
| C（可选） | off | on | on，独立后训 | 验证 novel-view AE/AWB |
| D（最后考虑） | 低容量、延迟 | on | off/on | 补 PPISP 无法覆盖的局部残差 |

除 appearance module 外，其余随机种子、训练步数、densification、LiDAR 约束、sky、pose/calibration 和数据范围都应保持一致。

实验 B 关闭 controller 后，held-out 图使用的是零 per-frame correction；它适合检查 raw canonical 表示和 per-camera 模块，但不能代表 PPISP 完整的 novel-view AE/AWB 能力。正式比较 novel-view PPISP 指标需要实验 C：先完成与基线等长的 scene + PPISP 训练，再额外冻结 scene、sky、pose/calibration、densification 与 PPISP 主参数，只训练 controller；不应把 controller 阶段硬塞进原 15k 步而缩短场景训练。

验收重点：

- 同一地面区域从不同视角渲染时，canonical RGB 的均值、方差和色差；
- 该区域高亮高斯的 SH0、opacity、厚度和离面距离是否减少异常相关性；
- raw canonical 与 PPISP-corrected 的训练视图对照；
- PPISP 学到的 exposure/color 曲线是否随帧平滑、是否与已有 metadata 一致；
- per-camera vignetting/CRF 是否合理且同相机跨序列稳定；
- 裸 PLY 在目标 viewer 中的观感；
- train-view 与 held-out/跨视角指标之间的 gap，而不只是 train PSNR。

## 工程风险

1. **CUDA 扩展依赖。** 官方实现是 CUDA-only fused forward/backward/regularizer，没有 CPU fallback；安装需按当前 PyTorch ABI 编译，官方建议 `--no-build-isolation`。[官方安装说明](https://github.com/nv-tlabs/ppisp#adding-ppisp-as-a-dependency)
2. **环境已完成最小验证。** `ppisp 1.2.1` 已安装到 `3dgeer` 环境；在 RTX 4090、PyTorch `2.11.0+cu128` 上完成了 fused forward、regularization 和 backward smoke test，输入及 exposure/vignetting/color/CRF 参数梯度均为有限值。尚未运行完整训练。
3. **依赖与许可。** `v1.2.1` 要求 Python >= 3.10，运行依赖 Torch、NumPy、Matplotlib；官方代码为 Apache-2.0。[pyproject](https://raw.githubusercontent.com/nv-tlabs/ppisp/v1.2.1/pyproject.toml)，[LICENSE](https://raw.githubusercontent.com/nv-tlabs/ppisp/v1.2.1/LICENSE)
4. **单图 API。** 官方 API 的 camera/frame index 要求单个标量，controller 接收单张 HWC 图。当前默认 batch size 1 匹配；扩大 batch 或多卡前需要额外适配。
5. **训练阶段语义。** 官方默认 controller distillation 会自动 detach radiance，但不会替调用方完整管理所有 Gaussian optimizer；直接照搬而不冻结 scene 会造成无效 step 或阶段混乱。
6. **多相机索引。** COLMAP camera id 不保证从 0 连续编号，不能直接作为官方参数张量下标。
7. **局部问题残留。** PPISP 的受限容量是泛化优势，也是当前局部高光/阴影问题的风险点；不能因“用了 PPISP”就假设局部变化已经被分离。
8. **通用 PLY viewer 不执行 PPISP。** 需要提前接受 raw canonical 外观，或为部署 viewer 增加 PPISP post-process。

## 最终判断

PPISP 与本项目“导出稳定 SH0 PLY，避免相机变化污染高斯颜色/几何”的方向一致，而且比 per-image bilateral grid 更有约束、更容易解释。它最可能改善全局 AE/AWB、多相机渐晕和 CRF 差异。

但针对用户最关心的局部脚下高光和阴影，PPISP 不是完整答案：真正的局部 tone mapping、移动阴影和视角相关镜面反射超出了它的模型范围。合理路径是先做 PPISP-only 对照，确认全局 ISP 被拆分后还剩多少局部问题，再决定是否加入低容量、延迟启用的 bilateral residual，或针对动态光照/高光设计 mask、鲁棒 loss 或显式反射模型。
