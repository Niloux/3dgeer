# LichtFeld Studio 深度约束方案调研

调研日期：2026-08-31

上游核对版本：LichtFeld Studio
[`de972c89ec6bcd27406f892b966f180a7054f2cc`](https://github.com/MrNeRF/LichtFeld-Studio/tree/de972c89ec6bcd27406f892b966f180a7054f2cc)

## 结论摘要

LichtFeld Studio 当前的深度监督不是 `geometry_enabled` 一类持续约束每个
Gaussian 的移植层，也不是简单的度量深度 L1。它把每张深度图当作可能只有相对
尺度的 **camera-space Z / disparity prior**，启动时用初始点云为每个相机拟合一次
`prior -> Z` 或 `prior -> inverse-Z` 的仿射变换，训练时再监督 alpha 归一化的
rendered expected Z。深度图提供稠密的逐像素监督，点云只负责一次性定标；后续
没有 per-Gaussian anchor、法向、邻域或 `geometry_enabled` 状态。

这对 3dgeer 的直接启示是：如果
`/home/wuyou/remote/preprocess/output/indoor` 中的图已经与训练 RGB/相机逐帧、逐像素
对齐，且能确认是 camera Z（或先从 ray distance 转成 Z），可以让外部深度图直接
成为几何监督来源，没必要仅为这项监督保留 `geometry_enabled`。但是否能直接接入
`simple_trainer.depth_loss`，仍取决于本地数据审计；尤其必须先确认文件匹配、尺寸、
数值编码、无效值和 Z-vs-range。

上游实现还有一个不能忽略的边界：深度 loss 目前只在 `gut=false` 的 FastGS 路径
执行，点云锚点也用针孔投影，没有传相机模型或畸变参数。因此 LichtFeld 的
**数据定标与鲁棒 loss 思路值得借鉴，但其代码不能原样作为 3DGUT/鱼眼支持的
依据**。

## 1. 深度图如何发现并匹配相机

### 目录、扩展名与最佳命名

COLMAP/Transforms 数据集根目录下会递归扫描 `depth/`，其次是 `depths/`；自动发现
接受 `.png`、`.jpg`、`.jpeg`、`.tif`、`.tiff` 和复合后缀 `.depth.png`。
[`filesystem_utils.hpp`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/io/include/io/filesystem_utils.hpp#L71-L83)

匹配键来自相机的 COLMAP `image_name`，匹配顺序可以概括为：

1. 相同相对路径/文件名（大小写不敏感）；
2. 保留相对目录和 stem，逐个尝试上述深度扩展名；
3. 若整个深度目录中 basename 唯一，允许 basename fallback；
4. 最后允许至少两位的尾部帧号匹配，例如
   `RENDER_0042.jpg -> DEPTH_0042.png`。

索引的 exact-relative-path 与 unique-basename 规则见
[`RecursiveFileCache`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/io/include/io/filesystem_utils.hpp#L185-L303)，
候选扩展和尾号 fallback 见
[`SidecarDirCache`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/io/include/io/filesystem_utils.hpp#L435-L530)。

因此最稳妥的约定是镜像 RGB 的相对目录并把扩展名换成 PNG，例如：

```text
images/room_a/frame_0042.jpg
depth/room_a/frame_0042.png
```

当同名候选不唯一且 depth loss 已启用时，加载器会把数据集判为 ambiguous，而不是
任选一个文件。相机创建时把解析到的 `depth_path` 直接绑定到对应 COLMAP image。
[`colmap.cpp`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/io/formats/colmap.cpp#L2588-L2603)
[`colmap.cpp`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/io/formats/colmap.cpp#L2684-L2703)

### 尺寸契约

深度图可以是 COLMAP 记录的原始相机尺寸，也可以是当前训练图尺寸的任意**等比例
整数倍**；宽高缩放倍率必须相同。这样一份原分辨率 sidecar 可以复用于
`images_2`、`images_8` 等训练图目录。
[`sidecar_dimensions_match_contract`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/io/include/io/filesystem_utils.hpp#L96-L116)
[`colmap.cpp`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/io/formats/colmap.cpp#L2641-L2655)

读取后若仍与 rendered depth 不同，trainer 会再用 Lanczos resize 到渲染尺寸；相机
走预去畸变路径时，depth 也会用相同的几何重采样处理。
[`camera.cpp`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/core/camera.cpp#L637-L657)
[`camera.cpp`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/core/camera.cpp#L693-L704)
[`trainer.cpp`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/trainer.cpp#L6692-L6711)

## 2. 文件格式、数值尺度与无效值

### 解码规则

- 大于 8 bit 的单通道图由 OpenImageIO 读取 channel 0 为 float32；16-bit 整数归一化
  到 `[0,1]`，float TIFF/EXR 保留浮点值。自动目录发现本身没有列出 `.exr`，所以
  常规 sidecar 应使用 PNG/TIFF。
- 8-bit 图走普通图像路径后除以 `255`；若读成 RGB，则三个通道取平均。
- 16-bit 文件的量化步长记作 `1/65535`，8-bit 为 `1/255`，float 为 `0`，后续用于
  loss 的量化 deadband。

源码分别见
[`image_io.hpp`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/core/include/core/image_io.hpp#L51-L67)、
[`image_io.cpp`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/core/image_io.cpp#L601-L629)、
[`camera.cpp`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/core/camera.cpp#L658-L691) 和
[`image_quantization_step`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/core/image_io.cpp#L663-L677)。

JPEG 虽然在允许列表中，但有损压缩会改变深度值；工程上应优先用单通道 16-bit
PNG 或 float TIFF。LichtFeld 自带 `preprocess` 的实际默认也是 16-bit PNG。
[`PreprocessParameters`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/core/include/core/parameters.hpp#L532-L547)
[`preprocess.cpp`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/preprocessing/preprocess.cpp#L1178-L1195)

### 无效值与 mask

loss 只接受同时满足以下条件的像素：

```text
target > 0
rendered alpha > 1e-3
target、depth accumulator、alpha 都是 finite
optional pixel weight > 0 且 finite
```

所以 `0` 是官方无效值；负数、NaN 和 Inf 也会跳过。没有单独的 depth-valid-mask
文件接口，mask 应编码为 target zero。该判断见
[`pixel_active`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/kernels/depth_loss.cu#L55-L75)。

深度项会按 rendered alpha 加权，并可再乘 trainer 的 ROI weight。当前调用传入的
是 `roi_weight`，不是普通 photometric `mask_tile`；因此不能假设 RGB segmentation
mask 会自动屏蔽深度 loss。
[`trainer.cpp`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/trainer.cpp#L6527-L6534)
[`trainer.cpp`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/trainer.cpp#L6735-L6763)

LichtFeld 自带 MoGe-2 预处理会取预测 3D points 的 `z`，在**每张图内部**把有效
`z` 的 min/max 线性映射到 `[0.02, 1]`，无效像素写 `0`，再写成 8/16-bit PNG。
因此官方生成物本身不是米制深度，而是需要相机级仿射定标的 relative Z prior。
[`build_depth_png`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/preprocessing/preprocess.cpp#L803-L847)

## 3. 深度含义：camera Z，不是沿 ray 的欧氏距离

FastGS 对每个 Gaussian 保存的 depth 是 world-to-camera 变换第三行作用于 3D mean
得到的 `z_c`；像素 depth accumulator 是按正常 alpha-compositing 权重累加的
`sum(T_i * alpha_i * z_i)`。
[`kernels_forward.cuh`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/rasterization/fastgs/rasterization/include/kernels_forward.cuh#L109-L115)
[`kernels_forward.cuh`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/rasterization/fastgs/rasterization/include/kernels_forward.cuh#L753-L785)

点云定标同样把 sparse point 变到相机后取 `z_c`，再用
`u=fx*x_c/z_c+cx, v=fy*y_c/z_c+cy` 采样 prior。
[`depth_anchor_collect_kernel`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/kernels/depth_loss.cu#L463-L500)

因此这里的几何目标是 optical-axis Z-depth，不是 camera center 到表面的 ray
distance/range。虽然 `ssi` 可以吸收整张图的 scale/shift，也能自动判断“数值越大越
远”还是“数值越大越近”，它不能用一个仿射变换消除 range 与 Z 之间随像素方向
变化的夹角因子。若外部图存的是单位 ray 上的距离 `r`，应先按相机模型转换为
`z = r * ray_cam.z`；鱼眼必须使用真实 unprojection ray，而不是针孔近似。

## 4. 初始点云定标：确实需要 sparse points，但不需要持续的 geometry state

### 每相机拟合

启动时，LichtFeld 把初始模型的 means 当作点云，投影到每张 prior，收集
`(target t, camera Z z)` 样本。它同时拟合两个候选：

```text
disparity-like:  1 / (z + f) = a_q * t + b_q
depth-like:      z             = a_z * t + b_z
f = 0.05 * median(z)
```

拟合采用 RANSAC 后最小二乘 refit；至少需要 256 个投影样本，inlier 数至少为原
样本的 30%，且 `|corr| >= 0.35`。prior 近乎常量也会被拒绝。
[`depth_loss.cu`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/kernels/depth_loss.cu#L435-L500)
[`depth_loss.cu`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/kernels/depth_loss.cu#L744-L823)

`ssi` 会在整个数据集上比较两类候选的累计 `|corr|`，统一选择 depth 或 disparity；
随后还会剔除没有该类拟合及 slope 符号与多数相机不一致的相机。也可以通过
`ssi-depth` / `ssi-disparity` 强制指定先验类型。
[`trainer.cpp`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/trainer.cpp#L427-L477)
[`Trainer::fitDepthAnchors`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/trainer.cpp#L2456-L2515)

### 点云是硬门槛

上游并不是“有 metric depth 就完全不需要点云”。当前 trainer 没有初始点时直接
不做 anchor；没有任何可靠相机 anchor 时整项 depth supervision 跳过。每步调用
loss 前也再次要求相机 anchor 已拟合且有效。
[`Trainer::fitDepthAnchors`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/trainer.cpp#L2382-L2406)
[`Trainer::fitDepthAnchors`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/trainer.cpp#L2529-L2537)
[`trainer.cpp`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/trainer.cpp#L6735-L6766)

这要求的是一个与相机位姿共坐标、足以投出可靠样本的初始 SfM 点云。没有数据集
点云时，LichtFeld 会用随机点初始化模型；随机点虽然让 model means 非空，但通常
无法通过上述相关性门槛，不能视为有效 depth anchor。
[`training_setup.cpp`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/training_setup.cpp#L260-L289)

### 不存在 `geometry_enabled` 式训练状态

上游的 depth 配置只有 `use_depth_loss`、`depth_loss_weight`、`depth_loss_mode`，默认
分别为 `false / 2.0 / ssi`。
[`parameters.hpp`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/core/include/core/parameters.hpp#L196-L200)

点云只产生一个按相机保存的 `DepthAnchor{model, scale, shift, floor, corr, samples}`；
结果可缓存为深度目录旁的 `depth_anchors.json`，key 是 camera `image_name`。它不会
随 densification 继承，也不参与 Gaussian split/prune。
[`depth_loss.hpp`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/kernels/depth_loss.hpp#L46-L67)
[`depth_anchor_cache.hpp`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/depth_anchor/depth_anchor_cache.hpp#L24-L62)
[`depth_anchor_cache.cpp`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/depth_anchor/depth_anchor_cache.cpp#L329-L354)

## 5. Loss 公式与权重

### 预测与 target 对齐

令一个像素的 rasterizer 输出为：

```text
D_acc = sum_i T_i alpha_i z_i
A     = sum_i T_i alpha_i
e     = max(D_acc, 0) / A              # alpha-normalized expected Z
p     = 1 / (e + f)                    # rendered softened inverse depth
```

源码在 primary/inverse statistics 中按 `A * pixel_weight` 计算 `e`、`p` 及其加权
标准差 `sigma_p`。
[`depth_loss.cu`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/kernels/depth_loss.cu#L112-L167)
[`depth_loss.cu`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/kernels/depth_loss.cu#L169-L249)

固定相机 anchor 把 prior 值 `t` 转成目标 inverse depth `d`：

```text
disparity prior: d = min(a*t + b, 1/f)
depth prior:     d = 1 / (a*t + b + f)
```

两种情况都要求拟合值 `a*t+b > 0`。
[`load_sample`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/kernels/depth_loss.cu#L251-L306)

### Robust data term + gradient term

它使用 Geman-McClure：

```text
rho(x) = 0.5 * x^2 / (1 + x^2)
db(r, delta) = sign(r) * max(|r| - delta, 0)
s = 2 * sigma_p
```

设有效像素权重 `omega_i = A_i * roi_weight_i`（无 ROI 时为 `A_i`），右/下相邻边
集合为 `E`，则 CUDA 实现对应：

```text
L_depth = lambda(k) / sum_i(omega_i) * [
    sum_i omega_i * rho(db(p_i - d_i, delta_i) / s)
  + lambda_grad * sum_(i,j in E) min(omega_i, omega_j)
      * rho(db((p_j-p_i) - (d_j-d_i), delta_i+delta_j) / s)
]
```

`lambda_grad=1`。量化 deadband 来自文件量化步长 `q`：先取
`h=0.5*q*|a|`；disparity prior 用 `delta=h`，depth prior 在反演后用
`delta=h*d^2`。这使落在 8/16-bit 半个量化台阶内的 residual 和梯度差不产生
loss/gradient，避免把平滑表面拉成量化阶梯。
[`depth_loss.hpp`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/kernels/depth_loss.hpp#L115-L141)
[`depth_loss.cu`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/kernels/depth_loss.cu#L35-L53)
[`depth_loss_grad_kernel`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/kernels/depth_loss.cu#L309-L432)
[`depth_loss_finalize_loss_kernel`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/kernels/depth_loss.cu#L631-L647)

loss 同时对未归一化 `D_acc` 和 alpha 输出梯度，所以它不仅移动 Gaussian 深度，
也会改变遮挡/透明度解释。
[`depth_loss_grad_kernel`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/kernels/depth_loss.cu#L422-L430)

### 时间权重

默认初始权重为 `2.0`，训练中按总 iteration fraction 指数衰减：

```text
lambda(k) = depth_loss_weight * 0.02 ^ min(k / total_iterations, 1)
```

所以默认约从 `2.0` 降到 `0.04`；gradient term 的内部权重固定为 `1.0`。深度标量
直接加到 photometric loss。
[`parameters.hpp`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/core/include/core/parameters.hpp#L196-L200)
[`trainer.cpp`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/trainer.cpp#L424-L425)
[`trainer.cpp`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/trainer.cpp#L6717-L6723)
[`trainer.cpp`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/trainer.cpp#L6748-L6766)

## 6. 对 3DGUT / 鱼眼移植的限制

当前 trainer 定义 `fastgs_path = !gut`，而 depth loss 的整个执行分支要求
`run_fastgs_gaussian_backward`；后者又要求 `fastgs_path`。因此 `gut=true` 时该分支
不会运行。
[`trainer.cpp`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/trainer.cpp#L5826-L5827)
[`trainer.cpp`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/trainer.cpp#L5872-L5886)
[`trainer.cpp`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/trainer.cpp#L6646-L6652)

此外，anchor collect API 只接收 `fx/fy/cx/cy`，投影 kernel 直接使用
`fx*x/z+cx, fy*y/z+cy`，没有 camera model、radial/tangential 或 fisheye 参数。
[`depth_loss.hpp`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/kernels/depth_loss.hpp#L69-L107)
[`depth_loss.cu`](https://github.com/MrNeRF/LichtFeld-Studio/blob/de972c89ec6bcd27406f892b966f180a7054f2cc/src/training/kernels/depth_loss.cu#L482-L500)

据此可作以下工程判断（这是基于源码的推论，不是上游已经验证的功能）：

- 逐像素 loss 本身可以移植到 distortion-aware renderer，只要 renderer 输出与
  target 同一像素坐标系中的 `D_acc`、alpha，并把两路梯度传回真实渲染路径；
- sparse-anchor 投影必须换成本项目相机模型的真实 project/unproject；
- 若外部 depth 已是可信 metric camera Z，可以直接固定单位监督，理论上无需上游
  这套 sparse affine anchor；若仍做 anchor，至少应只把它当一致性校验，而不应
  无条件重新标定可靠的米制尺度；
- 若外部图是 ray distance，则必须先转 Z，或让 renderer 与 target 都改成同一定义；
  不能一边用 Gaussian camera Z，一边直接监督 range。

## 7. `/home/wuyou/remote/preprocess/output/indoor` 的适配检查表

在决定删除 `geometry_enabled` 前，应对真实数据逐项得到明确答案：

| 检查项 | 可直接用于现有 Z-depth loss 的条件 | 不满足时的处理 |
| --- | --- | --- |
| 帧匹配 | 每个 training image 唯一对应一张 depth；最好按相对路径/stem 匹配 | 建立显式 image-name -> depth-path 映射，禁止按排序隐式配对 |
| 分辨率/坐标系 | 与训练 RGB 同视图、同 crop、同 resize、同畸变状态 | 用相同相机图像变换重采样，并同步处理 validity mask |
| 数值 dtype | float、16-bit 或可接受的 8-bit；解码后保留足够动态范围 | 明确 integer scale；避免把 16-bit 当 8-bit 或反之 |
| 无效值 | 能可靠生成 `valid = finite & (depth > 0)` | 把 sentinel（如 65535）显式转 mask，不能让它参加 loss |
| 几何含义 | optical-axis camera Z | ray distance 需按真实 ray 转 Z；disparity 需反演/SSI 对齐 |
| 尺度 | 可信 metric Z 可直接监督；relative prior 需每相机或全局对齐 | 借鉴上游 point-cloud affine anchor，并检查 corr/inliers |
| 相机模型 | target 与当前 fisheye/GUT renderer 使用相同 pixel ray | 不使用上游针孔 anchor kernel，改走本项目 distortion-aware 投影 |

只有前五项通过，才能回答“数据是否适配 `simple_trainer.depth_loss`”；第六项决定应
使用简单 metric loss 还是 SSI/anchor loss。无论哪一种，稠密 depth supervision
本身都不要求 `geometry_enabled` 的 per-Gaussian 迁移状态。

## 8. `indoor` 数据审计结果

本地数据已经满足直接接入 metric Z-depth loss 的条件：

- `images/`、`images_4/` 和 `depth/` 均为 732 帧；逐帧相对目录与 stem 完全一致，
  左右相机分别保留在 `L/`、`R/` 子目录中。
- 原图与 depth 均为 `3600 x 3600`；当前 `data_factor=4` 的训练图为
  `900 x 900`。depth 是单通道 uint16 PNG。
- 数据集 `report.json` 明确记录它是原始畸变像素域中的 camera-space Z，`0` 为
  invalid，编码为 `q = 1 + round(z / z_max * 65534)`，且 `z_max=40 m`。
- 全部分辨率每帧有效样本数为 63,980 到 648,193，中位数 487,502，没有低于
  LichtFeld 256-anchor 门槛的相机。抽检解码后的非零深度均为有限正数。
- `simple_trainer` 的 `RGB+ED` 输出是 alpha-normalized expected camera-space Z，
  因而 target 与 prediction 的几何量一致；当前训练保留鱼眼畸变，像素域也一致。

训练分辨率下不能对这种单像素稀疏 z-buffer 直接做普通插值，否则会把 invalid zero
混入深度。接入实现把所有 full-resolution 有效样本映射到对应训练像素，并在冲突时
保留最小 Z。首帧由 422,622 个 full-resolution 样本得到 161,716 个 `900 x 900`
有效监督像素，覆盖仍然充足。由于这些图已经是可信米制 Z，本项目直接解码后沿用
现有 inverse-depth L1；不采用 LichtFeld 为逐图相对深度设计的 affine anchor，也
不移植它只支持 FastGS/针孔路径的 CUDA loss。
