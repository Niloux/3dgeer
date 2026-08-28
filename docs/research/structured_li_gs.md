# Structured-Li-GS 调研与 3dgeer 几何约束重构建议

调研日期：2026-08-28

## 结论摘要

Structured-Li-GS 的正式题名是 **Structured-Li-GS: Structured 3D Gaussians
Splatting with LiDAR Incorporation and Spatial Constraints**，发表于 ISPRS Annals
2026。作者用 LiDAR、相机和 IMU 的手持/移动传感器数据，经 FAST-LIVO2 得到
位姿和稠密彩色点云，再以点云表面结构初始化并监督 3DGS。[正式论文与元数据](https://isprs-annals.copernicus.org/articles/XI-2-2026/375/2026/)，[作者实验室项目条目](https://cviss.net/research_cv/)

对 3dgeer 最重要的启发是：LiDAR 几何必须进入 Gaussian 的协方差，而不是继续
堆叠相互重叠的中心、法向、薄度、刷新、投影和剪枝规则。但本项目不采用论文的
固定 anchor topology，也不照搬它的 loss 组合。最终方案保留项目已有的 KNN-PCA
初始化、自由 Gaussian 参数化和标准 densification/pruning，只加入一个无状态、
协方差感知的局部平面距离。

> 来源状态：截至调研日，正式论文页、arXiv 条目和作者实验室页面都没有列出
> 官方代码仓库或补充材料。下文中论文没有交代的训练步数、学习率、offset 数量
> `k`、flatten factor `sigma_f` 等不能从官方实现核对；本文不会用猜测补齐。
> [arXiv 元数据](https://arxiv.org/abs/2606.27509)，[正式论文下载页](https://isprs-annals.copernicus.org/articles/XI-2-2026/375/2026/)

## 工程决策更新

讨论后的实现刻意不向 Structured-Li-GS 的整体结构对齐：

- 保留 KNN-PCA 初始化，但 `means/quats/scales` 在训练中仍完全自由；
- 保留项目原有的标准 densification 和 opacity/scale pruning；
- 不引入固定 anchor、local offset 参数化、LiDAR depth/normal loss 或 Poisson
  预处理；
- 删除动态 anchor state、CPU 最近邻刷新、surface-aware split/prune、三项独立
  几何 loss 和三类 hard projection；
- 只保留一个 `geometry_lambda`，控制单一的协方差感知 surfel loss。

对 Gaussian `N(mu, Sigma)` 和附近 LiDAR surfel `(p,n)`，实际使用：

```text
d_expected² = (nᵀ(mu-p))² + nᵀ Sigma n
L_geometry = weighted SmoothL1(d_expected / local_surface_scale)
```

第一项约束中心的法向偏移，第二项直接度量 Gaussian 沿 LiDAR 法向的方差；因此
一个式子同时表达“贴面”和“不要穿透表面”，不需要再判断哪个 scale 是最短轴，
也不需要单独的 quaternion alignment loss。surfel 关联由当前 Gaussian 位置每步
重新查询 GPU 稀疏 voxel index，不在 Gaussian 上保存可继承的 anchor ID。

下文保留论文事实和最初方案，便于说明取舍；涉及固定 topology/local-frame 的
内容是已放弃的备选方向，不是当前实现建议。

## 论文事实

本节只陈述论文明确写出的内容；对 3dgeer 的推断放在后续章节。

### 问题设置与传感器

论文处理的是由 LiDAR-inertial-visual SLAM 产生的 posed images 和稠密彩色点云，
目标是在室内、室外、弱纹理、走廊和大尺度场景中，以较少 Gaussian 获得较好的
新视角渲染。预处理采用 FAST-LIVO2 估计 LiDAR scan poses 并形成稠密点云。
[论文 3.2](https://arxiv.org/html/2606.27509#S3.SS2)

实验数据包括：

- FAST-LIVO2：MV-CA013-21UC 工业相机（1280x1024）、Livox Avia LiDAR 和
  IMU；
- HILTI22：手持平台为 10 Hz Hesai PandarXT-32、40 Hz 五相机（实验下采样到
  10 Hz）和 400 Hz BMI085；机器人平台为 10 Hz Robosense BPearl5、10 Hz
  八相机和 200 Hz MTi-670；论文只使用前向相机；
- 自采手持设备：MV-CA013-21UC、Livox MID360 和 NUC 11；相机 10 Hz，论文
  描述 LiDAR 与 GPRMC 消息以 1 Hz 触发，使用 Teensy timer 做硬件同步。

这些传感器与序列细节见[论文 4.1-4.2](https://arxiv.org/html/2606.27509#S4)。

### 数据预处理

FAST-LIVO2 输出之后，论文依次做点云 denoising、downsampling、normal
estimation，再做 Poisson Surface Reconstruction。随后按点到重建 mesh 的空间
距离进一步过滤 cleaned point cloud；同时把点云投影到各相机，生成 LiDAR depth
image `D_L` 和 normal image `N_L`。[论文 3.2](https://arxiv.org/html/2606.27509#S3.SS2)

这里有两个层次容易混淆：mesh 是**点云清理的中间工具**，最终训练表示仍然是
Gaussians；论文没有用 mesh 直接渲染，也没有报告 mesh-filter distance、法向估计
邻域或 Poisson 参数。

### 几何表示与初始化

论文仍使用各向异性 3D Gaussian，协方差为

```text
Sigma = R S S^T R^T
```

颜色使用 spherical harmonics，渲染为标准投影和前向 alpha compositing。
[论文 3.1](https://arxiv.org/html/2606.27509#S3.SS1)

输入点云 `P` 以 voxel size `epsilon` 去重：

```text
P' = { p_i in P | floor(p_i / epsilon) is unique }
```

每个保留点作为 anchor。论文把 anchor 写成带 32 维 local context feature
`f_v`、三维 scaling factor `l_v` 和 `k x 3` learnable offsets `O_v` 的结构，明显
沿用了 Scaffold-GS 的 anchor 术语；但论文没有给出这些 feature 如何解码为颜色、
opacity、rotation/scale，也没有给 `k` 的值。[论文 3.3.1](https://arxiv.org/html/2606.27509#S3.SS3.SSS1)

局部表面法向 `n` 用于初始化旋转。论文以初始主轴 `v_1=(0,0,1)`，取
`a = v_1 x n`、`theta = acos(v_1 dot n)`，再构造 quaternion

```text
q = (cos(theta/2), sin(theta/2) a_x,
     sin(theta/2) a_y, sin(theta/2) a_z),
```

使 `q v_1 q^-1 = n`。随后只压缩法向轴 `s_3`：`s_3 = sigma_f * s_3`。
[论文 3.3.2](https://arxiv.org/html/2606.27509#S3.SS3.SSS2)

论文把 `s` 同时写成 log scale、又直接乘 flatten factor，因此单凭公式无法确定
实现是在 log space 还是 physical scale 上压缩；`sigma_f` 也没有报告。这一部分
只能借鉴“按局部表面法向旋转并压薄”的思想，不能逐公式复现。

### LiDAR/点云监督损失

总损失写为

```text
L_G = lambda_c L_RGB + lambda_f L_f + lambda_o L_o
    + lambda_D L_D + lambda_N L_N.
```

五项的原文定义如下。[论文 3.4](https://arxiv.org/html/2606.27509#S3.SS4)

1. Photometric loss：

   ```text
   L_RGB = (1-lambda) L1(I_render, I_gt)
         + lambda L_D-SSIM(I_render, I_gt),  lambda=0.2.
   ```

2. Flatten loss：`L_f = ||s_3||_2`，即惩罚论文所定义的法向轴 scale。

3. Offset loss：

   ```text
   L_o = lambda_o3 ||o_3||_1
       + lambda_o12 ||o_1 * o_2||_1,
   lambda_o3=5, lambda_o12=1.
   ```

   论文称 `o_3` 为法向 offset，`o_1,o_2` 为切平面 offset，因此对法向施加更强
   约束。[论文 3.4.3](https://arxiv.org/html/2606.27509#S3.SS4.SSS3)

4. Depth loss：把点云投影深度 `D_L` 与 rendered depth `D_render` 转为 inverse
   depth 后做 masked L1：

   ```text
   M_D(i,j) = 1  if T_min < D_L(i,j) < T_max, else 0
   L_D = L1(inv(D_render), inv(D_L)) * M_D.
   ```

   论文没有报告 `T_min/T_max`、mask reduction 或遮挡冲突细节。
   [论文 3.4.4](https://arxiv.org/html/2606.27509#S3.SS4.SSS4)

5. Normal loss：论文最终使用的不是 Gaussian 法向与 LiDAR 法向的逐点 cosine
   alignment，而是 rendered normal 的 edge-aware smoothness：

   ```text
   L_N = (grad(N_render) * w(E)) * M_N,
   w(x) = (x-1)^q, q=400.
   ```

   `E` 是由 3D LiDAR point cloud 提取的 edge image，`M_N` 屏蔽无 normal 的
   像素。TeX 源中曾写过 `1 - N_L dot N_render` 的 absolute normal loss，但这段
   在正式公式中被注释掉，最终论文只有上述 normal consistency loss。
   [论文 3.4.5](https://arxiv.org/html/2606.27509#S3.SS4.SSS5)，[正式 PDF](https://isprs-annals.copernicus.org/articles/XI-2-2026/375/2026/isprs-annals-XI-2-2026-375-2026.pdf)

因此，Structured-Li-GS **没有论文正文中的动态 nearest-surface point-to-plane
loss**，也没有当前 3dgeer 那种“Gaussian 最短轴 vs anchor normal”的 per-Gaussian
法向 loss。它用固定 anchor offset、flattening、图像空间 depth 和 edge-aware
rendered-normal smoothness来约束几何。

### Densification、pruning 与训练阶段

论文反复明确写出“without Gaussian densification”和“maintaining a fixed number
of Gaussians”。CBD2 的初始点数和最终 Gaussian 数都为 356,885，也与固定拓扑的
描述一致。[论文摘要/方法动机](https://arxiv.org/html/2606.27509#S1)，[Gaussian 数量比较](https://arxiv.org/html/2606.27509#S5.SS4)，[结论](https://arxiv.org/html/2606.27509#S6)

论文没有描述独立的 geometry warmup、分阶段冻结或后处理阶段，也没有给训练
iterations/epochs。最稳妥的解读是全部已列出的 loss 联合训练，而不是假设存在
隐藏阶段。[论文 3.4](https://arxiv.org/html/2606.27509#S3.SS4)

论文没有描述 opacity pruning、surface pruning 或 densification 后的 anchor
继承。由于作者强调 fixed number，且 CBD2 的最终数量精确等于表中的初始点数，
证据与“不 grow、不 prune”的固定拓扑一致；但没有官方代码，无法排除不改变总数
的内部替换策略，不能把这一点说得比论文更强。

### 关键超参数与未报告项

| 项目 | 论文明确值 | 来源 |
| --- | ---: | --- |
| RGB D-SSIM weight | `0.2` | [3.4.1](https://arxiv.org/html/2606.27509#S3.SS4.SSS1) |
| 法向/切向 offset 内部权重 | `5 / 1` | [3.4.3](https://arxiv.org/html/2606.27509#S3.SS4.SSS3) |
| normal edge exponent `q` | `400` | [3.4.5](https://arxiv.org/html/2606.27509#S3.SS4.SSS5) |
| 主实验点云 voxel | `0.06 m` | [5.1](https://arxiv.org/html/2606.27509#S5.SS1) |
| FAST-LIVO2/HILTI 图像下采样 | `5x` | [5.1](https://arxiv.org/html/2606.27509#S5.SS1) |
| 自采数据图像下采样 | `3x` | [5.1](https://arxiv.org/html/2606.27509#S5.SS1) |
| test split | 每第 8 帧 | [5.1](https://arxiv.org/html/2606.27509#S5.SS1) |
| `lambda_c,lambda_o,lambda_D,lambda_N` | 均为 `1` | [5.1](https://arxiv.org/html/2606.27509#S5.SS1) |
| ablation 点云/Gaussian voxel | `0.0065 m / 0.005 m` | [5.5](https://arxiv.org/html/2606.27509#S5.SS5) |

论文没有报告 `lambda_f`，也没有报告 optimizer、learning rates、训练步数、`k`、
`sigma_f`、depth thresholds、Poisson 参数、normal 邻域、edge extraction 参数和
各 loss 的 reduction/normalization。主实验写 0.06 m voxel，而 CBD2 ablation 又写
0.0065/0.005 m，尺度相差约一个数量级且没有解释两种 voxel 的关系。这些都使得
精确复现依赖尚未公开的实现。[论文实验设置与消融](https://arxiv.org/html/2606.27509#S5)

### 定量结果与消融

CBD2 上的模型大小比较为：

| 方法 | Gaussian 数量 |
| --- | ---: |
| 3D-GS | 1,893,221 |
| 2D-GS | 929,169 |
| Scaffold-GS | 521,782 |
| LetsGo | 1,005,452 |
| AtomGS | 2,005,056 |
| Structured-Li-GS | **356,885** |

该序列中 Structured-Li-GS 用固定的 356,885 个 Gaussian 得到 22.82 PSNR、
0.760 SSIM、0.389 LPIPS；其主要优势是以最少 primitive 达到与最佳方法相当或
更好的渲染指标，而不是所有序列上绝对最优。[论文 Table 2 与 Table 4](https://arxiv.org/html/2606.27509#S5)

CBD2 单场景 loss ablation 为：

| 配置 | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| w/o flatten | 22.72 | 0.7574 | 0.3942 |
| w/o offset | 22.77 | 0.7568 | 0.3969 |
| w/o depth | 22.75 | 0.7584 | 0.3927 |
| w/o normal | 22.71 | 0.7576 | 0.3926 |
| all | **22.82** | **0.7601** | **0.3896** |

每项带来的 PSNR 增量只有约 0.05-0.11 dB，而且论文没有报告重复运行方差。
正文声称 RGB-only 为 22.07 dB/0.7406 SSIM，但这行在最终表格中没有展示；没有
初始化 ablation、fixed-topology ablation 或几何误差指标。[论文 5.5](https://arxiv.org/html/2606.27509#S5.SS5)

## 论文证据的边界

以下不是否定方法，而是决定“哪些可以迁移、哪些不能照抄”时必须保留的边界：

1. **只报告渲染质量，没有几何指标。** PSNR/SSIM/LPIPS 不能证明表面 Chamfer、
   normal error 或 mesh completeness 更好；论文展示了 depth/normal render，未
   对它们做定量评估。[实验指标定义](https://arxiv.org/html/2606.27509#S5)
2. **loss ablation 很窄。** 只有 CBD2 一个序列，差异小，没有均值/方差，也没有
   检查各种 loss 对最终 Gaussian 数、离面误差和厚度的影响。
3. **核心实现信息不完整。** anchor feature、`k` offsets 和 Gaussian 属性之间的
   映射没有公式；关键优化设置与 `lambda_f` 缺失；官方代码未公开。
4. **normal loss 名称容易造成误读。** 最终公式是渲染法向 smoothness，而不是
   LiDAR normal alignment；直接把当前 `surface_normal_lambda` 称作复现论文并不
   准确。[论文 3.4.5](https://arxiv.org/html/2606.27509#S3.SS4.SSS5)
5. **offset product loss 有退化自由度。** `|o_1 o_2|` 在任一切向分量为零时不会
   约束另一个分量。论文没有解释为何不用 `|o_1|+|o_2|` 或二维范数，因此不应
   机械照搬。[论文 3.4.3](https://arxiv.org/html/2606.27509#S3.SS4.SSS3)
6. **预处理质量是前提。** 作者在 future work 中明确说还需加强 surface normal
   estimation 和 depth image generation；这说明错误法向/深度会直接限制方法。
   [论文结论](https://arxiv.org/html/2606.27509#S6)
7. **Poisson 不是普适真值。** 论文没有消融 mesh filtering。对植被、细杆、开放
   表面或法向未定向点云，Poisson 可能闭合空洞或删除真结构；只能作为可验证的
   清理选项，不能当作 3dgeer 的强制步骤。

## 与重构前 3dgeer 的逐项对照

### 已经做对、应保留的部分

当前 KNN-PCA 初始化已经比论文文字描述更具体：它对局部 covariance 做特征分解，
以最小特征向量作 normal，以两个切向特征值设置椭圆尺度，按 planarity/curvature
决定是否采用 surface-aligned 初始化，并把 normal scale 压到切向 base scale 的
固定比例。[当前 KNN-PCA 实现](../../examples/gaussian_models.py#L85)

这与 Structured-Li-GS 的 normal-assisted rotation/flattening 是同一核心思想，
没有必要推倒重写。现配置的 `init_use_knn_pca: true`、`k=24`、normal factor
`0.25` 可以继续作为固定拓扑基线，但阈值和尺度应根据 Park 点云的局部密度做一次
分布统计，而不是从论文的 0.06 m 盲抄。[当前配置](../../configs/simple_trainer/simple_trainer.yaml#L37)

### 重构前复杂度来自哪里

当前 `LidarSurfaceGeometry` 在每个 Gaussian 上维护 anchor point、anchor normal、
confidence、support radius、valid flag、离面计数和 prune mask；训练中计算
point-to-plane、短轴 normal alignment 和厚度 ratio 三项 loss。
[anchor state 与 loss](../../examples/lidar_geometry.py#L147)

为了支持 densification 后的新 Gaussian，它还周期性把位移较大的 Gaussian 搬到
CPU 做 nearest-neighbor reassociation，并维护 persistent off-surface pruning。
[refresh/prune](../../examples/lidar_geometry.py#L601)

训练器又在 warmup 后可选执行 center、thickness 和 tilt 三类 hard projection；
配置同时保留 loss 权重、dead zone、distance scale、最大距离、confidence、刷新
周期/批量、各类 hard-bound threshold 和 prune patience。
[hard projections 调用](../../examples/simple_trainer.py#L2410)，[当前参数组](../../examples/simple_trainer.py#L469)

当前 YAML 一共暴露 31 个 `surface_*` 字段，而且 dataclass 默认值与 YAML 在
surface/normal/thickness 权重、loss 采样上限、tilt 角度和 prune 默认开关上都不
一致。这意味着调用方不仅要理解 31 个旋钮，还要理解“未加载 YAML 时是另一套
行为”；它已经是一个浅模块 interface，而不只是参数命名显得繁琐。
[dataclass 默认值](../../examples/simple_trainer.py#L469)，[YAML 值](../../configs/simple_trainer/simple_trainer.yaml#L48)

这些机制本身并非没有理由，但它们是在解决同一个结构性冲突：**Gaussian 允许
任意增长和移动，同时我们又希望它们保持一一对应的 LiDAR surface support。**
Structured-Li-GS 直接固定 topology，因此根本不需要解决 densification 后的
anchor 归属问题。

### 关键差异表

| 维度 | Structured-Li-GS（论文事实） | 当前 3dgeer | 判断 |
| --- | --- | --- | --- |
| 初始点 | voxelized dense colorized LiDAR | LiDAR PLY | 同方向 |
| 初始法向/尺度 | normal-assisted rotate + flatten | KNN-PCA rotate + anisotropic scale | 当前更具体，可保留 |
| topology | fixed count，无 densification | 3001-27000 step densify，最多 9M | 最大分歧 |
| center 约束 | anchor-local offsets，法向更强 | 动态 point-to-plane + hard clamp | 可由局部参数化简化 |
| 法向约束 | edge-aware rendered-normal smoothness | Gaussian 短轴对齐 anchor normal | 不是同一个 loss |
| depth | 投影稠密点云 depth image | 当前 depth path 投影 COLMAP visible points，做 sparse disparity sample | 应改成 LiDAR depth cache/投影 |
| thickness | init flatten + `||s_3||` | shortest/middle ratio loss + hard ratio | 可改成结构化 scale |
| anchor refresh | 未描述/固定拓扑不需要 | 周期 CPU NN query | 可删除 |
| surface prune | 未描述，结果固定数 | persistent off-surface prune | 首轮删除 |
| hard projection | 未描述 | offset/thickness/tilt 三套 | 首轮删除 |

当前 sparse depth 路径确实在 rendered expected depth 上查询投影点并比较 disparity，
但数据来自 `parser.points` 和每图 visible point indices，不是论文的稠密 LiDAR
depth image。[当前 depth 数据生成](../../examples/datasets/colmap.py#L677)，[当前 depth loss](../../examples/simple_trainer.py#L1942)

## 初版迁移设想（讨论后未采用）

本节记录最初更贴近论文的方案，仅作为设计取舍背景；当前实现以“工程决策更新”
一节为准。

### 直接借鉴

1. **固定 topology 的首轮实验。** 关闭 densification，不在训练中创建/分裂
   Gaussian；primitive 数由 voxelized LiDAR coverage 决定。这是论文模型小且
   约束简单的主要结构来源，不是某个 loss weight 的功劳。
2. **一 anchor 一局部表面坐标系。** 每个 anchor 保存两个 tangent 和一个 normal，
   center/scale 都在该坐标系参数化；当前 KNN-PCA 已经产生所需 basis。
3. **真正的 LiDAR image-space depth supervision。** 用当前鱼眼投影模型和相机位姿
   对 cleaned LiDAR 做 z-buffer，生成 valid mask 和 metric depth；用 inverse depth
   robust loss 监督 renderer。它比全局最近邻 point-to-plane 更直接地约束实际可见
   表面和遮挡顺序。
4. **把数据清理当成方法组成，而不是 loss 补丁。** 先排除孤立点、多路径/动态
   物体和明显离群 surface，再创建 anchors；不要指望训练期 surface pruning 修复
   错误 PLY。
5. **以固定模型大小作为主要验收维度。** 除渲染指标外，报告 Gaussian 数、显存、
   离面误差、depth error 和 normal error，避免只看 training PSNR。

### 需要改造后借鉴

1. **Offset loss。** 保留论文“法向比切向强”的思想，但不用退化的
   `|o_1 o_2|`。建议使用 local-frame Huber/L1：

   ```text
   L_anchor = mean_i w_i [ 5 huber(delta_n / sigma_n)
                         + huber(||delta_t||_2 / sigma_t) ].
   ```

   `5:1` 可以作为论文启发的固定内部比例，用户只需要一个
   `geometry_anchor_lambda`，不再暴露 dead zone、distance scale、max distance、
   confidence threshold 和 refresh threshold 的组合。
2. **Flattening。** 不直接最小化可能是 log scale 的 `s_3`。把 normal scale 绑定
   为切向尺度几何均值的比例，或使用有下界/上界的 physical ratio；首轮把该比例
   固定成当前已经验证过的 `0.25`，不需要 thickness loss 和 hard bound。
3. **Normal supervision。** 论文的 edge-aware smoothness 适合大平面，但可能抹掉
   树枝、路沿和薄物体。先用结构化 normal 参数化保证每个 Gaussian 的 normal，
   LiDAR rendered-normal loss只作为第二阶段可选项；如加入，优先直接 cosine
   alignment 加 edge mask，而不是只做 smoothness。
4. **Poisson filtering。** 做成离线可选 profile，并保存 before/after coverage、
   point-to-mesh distance 分布和被删点可视化。Park 类室外数据不应默认启用。

### 不宜照搬

- 不照搬论文的全局 `lambda=1`：各实现的 loss reduction、图像分辨率、scene
  scale 和有效像素比例不同，而且论文连 `lambda_f` 都没有报告。
- 不照搬 0.06 m voxel：它对应论文自己的传感器密度和下采样分辨率。3dgeer 应
  按目标 Gaussian 数、点间距分位数和相机 GSD 选择 voxel。
- 不照搬 `q=400` edge function：论文没有给 edge normalization，`(x-1)^400`
  对输入范围极敏感。
- 不引入论文的 32-D anchor feature/MLP：论文缺少解码公式和代码，当前直接 SH
  Gaussian 更透明，也更适合先验证几何假设。
- 不把论文当成硬约束有效性的强证据：它没有 geometry metrics，loss ablation
  增益很小且只在一个场景上报告。

对当前 Park 数据的只读统计进一步说明了这一点：`lidar.ply` 有 1,488,579 个点，
范围约为 `88.3 x 90.5 x 24.6 m`，PLY header 只有 XYZ/RGB、没有 normals。按规则
grid 做去重时，不同 voxel size 得到的 anchor 数为：

| voxel size | Park anchor 数 |
| ---: | ---: |
| 0.02 m | 1,468,388 |
| 0.03 m | 1,411,516 |
| 0.05 m | 1,178,747 |
| 0.06 m | 990,109 |
| 0.10 m | 506,916 |

所以照搬论文的 `0.06 m` 在 Park 上仍接近一百万 Gaussian；若目标接近论文 CBD2
的 35.7 万量级，需要更粗的 voxel、目标数量驱动的尺度搜索，或密度自适应采样，
而不是把 `0.06` 当作方法常数。

## 未采用的 `structured_lidar_v0` 备选方案

下面是面向 3dgeer 的设计推断，不是论文原样复现。目标是用更少的概念获得比当前
`surface_*` 更可解释的约束。

### 1. 离线表面资产

对 `lidar.ply` 一次性完成：

```text
raw LiDAR
  -> dynamic/outlier filtering
  -> voxel downsample
  -> KNN-PCA basis + confidence
  -> optional mesh-consistency filtering
  -> per-camera fisheye z-buffer depth/normal/mask cache
  -> fixed anchors
```

缓存中应记录生成时的 camera pose/intrinsics hash。当前项目会优化 pose/calibration，
如果 camera 参数还会变化，不能继续使用旧的 LiDAR projection：要么在 camera
阶段结束后重新生成 cache，要么训练中按当前相机动态投影。

### 2. 结构化 Gaussian 参数

对 anchor `p_i` 和 KNN-PCA frame `B_i=[t_i1,t_i2,n_i]`，把 center 写成

```text
mu_i = p_i + B_i [delta_u, delta_v, delta_n]^T.
```

首轮固定 Gaussian normal 为 `n_i`，只学习切平面内 yaw：

```text
R_i = B_i R_z(psi_i).
```

学习两个 tangent log-scales `a_i,b_i`，normal scale 用一个全局比例 `rho` 绑定：

```text
s_u = exp(a_i)
s_v = exp(b_i)
s_n = rho * sqrt(s_u s_v),  rho=0.25 initially.
```

这样天然满足 normal alignment 和 surface thickness，直接删除
`surface_normal_*`、`surface_tilt_*`、`surface_thickness_*` 全部训练参数。若后续
发现曲面细节确实需要 tilt，再增加一个受限的两维 tangent rotation residual，而
不是一开始恢复完整 quaternion + soft loss + hard cone。

### 3. 最小损失

```text
L = L_RGB + lambda_anchor L_anchor + lambda_depth L_inv_depth.
```

- `L_RGB` 沿用项目当前 photometric/PPISP 设计；
- `L_anchor` 是上述 local offset robust penalty，法向/切向内部比例固定 5:1；
- `L_inv_depth` 只在 z-buffer valid、非边界、非动态 mask 像素上计算 masked Huber
  inverse-depth error。

这使 geometry 只有两个可调权重。首轮不加 normal smoothness；只有当 depth
对平面噪声敏感、且 depth/normal 几何指标证明有收益时，再加一个可选
`lambda_normal`。

### 4. 两阶段训练

Structured-Li-GS 本身没有报告阶段；这里是针对 3dgeer 同时优化 pose、fisheye
calibration 和 appearance 的工程适配：

1. **Camera/appearance stabilization**：固定 anchor geometry，不 densify；优化
   pose/calibration 与必要的 photometric adapter。
2. **Structured surface optimization**：冻结或显著降低 camera 参数学习率，按最终
   camera 重建 LiDAR depth cache；解锁 local offsets、tangent scales、opacity 和
   SH，联合训练最小损失，仍不 densify。

如果输入 SLAM pose/calibration 已经足够可信，可以跳过第一阶段。若保留 pose
在线更新，则 LiDAR projection 必须同步更新，否则 depth supervision 与当前相机
不一致，会把标定误差写进 Gaussian geometry。

### 5. 建议配置面

首版只暴露：

```yaml
geometry_mode: structured_lidar
geometry_voxel_size: null          # null = use prepared PLY as-is
geometry_normal_scale_ratio: 0.25
geometry_anchor_lambda: 0.10
geometry_depth_lambda: 0.01
geometry_unlock_step: 1000
geometry_densification: false
```

`5:1` offset ratio、robust loss delta、mask erosion 等先设为实现常量并在文档中说明；
只有消融证明跨数据集必须调时才提升为配置。当前三十余个 `surface_*` 配置可以整体
进入 deprecated path，而不是一对一重命名。

### 6. 可选的后续增长

固定 topology 若确实欠拟合，只增加一个受控实验：允许在高 residual anchor 上沿
切平面 split，child 继承同一个 anchor frame，禁止法向随机位移。这样仍不需要
CPU nearest-neighbor refresh。只有当该方案在 held-out rendering 和 geometry
指标上都优于提高 LiDAR voxel density，才值得加入主线。

## 建议实验顺序与验收

保持随机种子、图像、PPISP/sky、pose/calibration 设置一致，至少比较：

| 实验 | KNN-PCA init | densify | anchor loss | LiDAR depth | 目的 |
| --- | --- | --- | --- | --- | --- |
| A 当前基线 | on | on | off | 当前 sparse depth/off | 已知上限 |
| B fixed init | on | **off** | off | off | 单独测固定 topology |
| C structured anchor | on | off | on | off | 测局部 anchor 参数化 |
| D structured + depth | on | off | on | **dense LiDAR** | 完整 v0 |
| E optional normal | on | off | on | dense LiDAR | 只在 D 有明确残差时测 |

不能只看 training PSNR。至少记录：

- held-out PSNR/SSIM/LPIPS；
- Gaussian 数、checkpoint 大小、峰值显存和 render FPS；
- Gaussian center 到 cleaned LiDAR 的 point-to-plane p50/p90/p95；
- held-out rendered depth 的 inverse-depth MAE 和 metric depth RMSE；
- Gaussian normal 与可靠 LiDAR normal 的角度 p50/p90；
- 空洞、薄物体、植被、天空边界和近地面区域的分区指标/可视化；
- camera/geometry 联合优化时 depth cache 与当前相机参数是否一致。

进入主线的最低条件应是：D 相对 B/C 显著降低 held-out depth/离面误差，且渲染
指标没有超出可接受幅度；否则说明问题在 LiDAR cleaning、pose/calibration 或
projection，而不是需要增加更多几何 loss。

## 最终判断

Structured-Li-GS 证明了手持 LiDAR-camera 数据上的表面先验有价值，但没有证明
3dgeer 应采用它的固定拓扑、anchor network 或整套图像空间监督。对本项目真正
可迁移的是“初始化时建立可靠局部表面、训练时直接约束 Gaussian 几何”的原则。

因此当前落地方案是 KNN-PCA 初始化加一个无状态、协方差感知的局部 surfel loss；
Gaussian 的表达能力、增密和标准剪枝全部保留。配置从 31 个 `surface_*` 字段缩减
为 `geometry_enabled` 与 `geometry_lambda`。若这一最小方案仍不能改善独立几何
指标，应优先检查 LiDAR/Pose/标定和 surfel 质量，而不是重新堆叠 hard bound。
