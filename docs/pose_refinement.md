# 联合位姿优化

`examples/simple_trainer.py --pose-opt` 使用
[GloSplat 第 3.3 节](https://arxiv.org/html/2603.04847v1#S3.SS3)中的联合训练目标：

`loss = photometric_loss + pose_opt_ba_lambda * track_reprojection_loss`

其中，`photometric_loss` 为光度损失，`track_reprojection_loss` 为特征轨迹的重投影损失。

每张训练图像都有独立的相机局部刚体位姿修正，包括帧编号相同的 L/R 图像。
旋转采用 SO(3) 切空间向量参数化。参考训练图像的位姿保持固定，用于锚定坐标系。
相机内参和畸变参数保持固定，渲染器与重投影损失使用同一组修正后的相机位姿。

数据加载器读取 COLMAP 中真实的 `points2D` 观测及其对应的三维特征轨迹 ID。
只有被至少两张训练图像观测到的三维点才会进入可优化轨迹点表，验证视图不提供观测约束。
轨迹点表独立于高斯中心，不受高斯增密、重定位和剪枝影响，也独立于 LiDAR 高斯初始化。

二维观测会转换到当前图像分辨率，按需去畸变，并根据 ROI 和随机裁剪调整坐标。
图像范围外及被掩码排除的观测会被丢弃。重投影支持针孔模型、径向/OpenCV 畸变及
`OPENCV_FISHEYE`。相机坐标系中深度非正的点不参与当前步骤的重投影损失。
图像或裁剪区域没有有效观测时，BA 损失为零；整个数据集缺少多视图 SfM 特征轨迹时，
程序会报错，不会退回仅使用光度损失的位姿优化。

```yaml
pose_opt: true
pose_opt_lr: 1.0e-5
pose_opt_ba_lambda: 1.0e-4
pose_opt_huber_delta: 1.0
pose_opt_track_lr: 1.6e-4
pose_opt_reference_image_id: 0
```

相机 Adam 学习率、BA 权重和 Huber 阈值采用论文中的取值。
Huber 损失作用于二维残差的欧氏范数，阈值单位为当前训练图像的像素。
每张图像内的残差损失求和后，再对小批量中的图像取平均；采样图像中的所有有效观测均参与计算。
改变图像分辨率会改变像素误差的尺度。

论文未明确给出轨迹点学习率及全部优化器细节。本实现为轨迹点使用独立的 Adam 优化器，
学习率为 `pose_opt_track_lr * scene_scale`，默认基础学习率为 `1.6e-4`。
位姿和轨迹点学习率均保持恒定，不使用权重衰减、位姿先验、预热或提前冻结，
从第一个训练步骤持续优化到最后一步。固定参考位姿和小批量损失归约方式是本实现的明确选择。
现有 SfM 预处理及高斯训练设置保持可用；这里实现的是论文中的联合优化部分，
未引入 GloSplat 的特征匹配与全局 SfM 流程。

训练日志包含 `pose/ba_loss`、`pose/weighted_ba_loss`、
`pose/reprojection_error_px` 和 `pose/track_observations`，分别记录原始 BA 损失、
加权 BA 损失、平均像素重投影误差及有效轨迹观测数。
检查点同时保存 `pose_adjust` 和 `track_adjust`，后者包含轨迹点坐标及原始点索引。
训练视图评估使用修正后的位姿；留出的验证视图保持输入位姿，不进行测试时优化。

使用旧版启动配置时，请删除以下已移除字段：`rig_opt`、
`rig_reference_camera_id`、`rig_reference_frame_id`、`pose_opt_start_step`、
`camera_freeze_step`、`pose_opt_translation_lr`、`pose_opt_rotation_lr`、
`pose_opt_rotation_mode`、`pose_opt_reg`、`pose_opt_prior_lambda`、
`pose_opt_translation_sigma` 和 `pose_opt_rotation_sigma_deg`。
旧版 rig/6D 位姿检查点不做迁移。联合位姿优化检查点必须包含轨迹状态，
并使用相同的 SfM 模型和训练集划分进行评估。

## 后续实验计划（2026-09-10，待实施）

结合 [Energy-GS 笔记](Energy-GS_pose_optimization_notes.md) 的讨论，保留现有
GloSplat 联合 RGB + BA 目标和独立 SfM 轨迹点，分两次改动验证：先限制早期高斯
自由度，再加入由粗到细的图像监督。以下为下次工作的计划，尚未实现或运行对照实验。

### 第一次改动：可开关的早期对齐阶段

- 首轮暂定前 `2000` 步，并根据训练图像数和 batch size 调整，保证完整遍历训练集。
  这是实验起点，不是已经验证的最佳阶段长度。
- 固定高斯中心：跳过 `means` 的优化器更新，保留反向传播并正常清理梯度。
- 暂停 MRNF 增密、剪枝等结构变化；实现时确认策略内部没有其他中心更新绕过冻结。
- Pose 和 SfM 轨迹点从第一步继续通过 RGB + BA 联合优化；颜色、透明度、尺度、
  旋转等其他高斯属性沿用现有学习方式。
- 对齐阶段结束后恢复现有高斯训练流程。该阶段可关闭，以保留当前方案作为对照。

本轮验证的问题：利用 LiDAR 初始几何，限制高斯中心移动和拓扑变化，是否能让图像
错位更多地通过 pose 修正，而不是由场景重新拟合。

### 第二次改动：对齐阶段的渐进模糊监督

- 对渲染图和 GT 同时应用 Gaussian blur，在对齐阶段逐渐减弱模糊，最终恢复原始监督。
- 保持图像尺寸、相机内参、二维观测和 BA 像素单位不变，以隔离监督难度的影响。
- 模糊需要正确处理鱼眼无效区域及其他 mask，避免边界外的零值污染有效像素。
- 初始模糊强度和衰减曲线留待实现时确定；这是基于 Energy-GS 的工程实验，
  不等同于其 SVD 低秩 GT 监督，也尚无效果结论。

### 对照设置与判断标准

| 版本 | 改动 |
| --- | --- |
| A | 当前 GloSplat 联合优化方案 |
| B | A + 早期对齐阶段 |
| C | B + 渐进模糊监督 |

三组使用相同的数据、初始化、训练预算、pose 学习率、BA 权重及曝光配置。
讨论时的 [展厅配置](../configs/simple_trainer/exhibition_hall.yaml) 为 `15000` 步，
`pose_opt_lr: 1e-4`、`pose_opt_ba_lambda: 1e-4`，增密第 `500` 步放行，
局部曝光网格第 `1000` 步开启。第一轮不同时修改曝光时序或 pose 学习率；
论文及代码默认的 `1e-5` 与当前 `1e-4` 的学习率比较单独进行。

- 优先观察重投影误差、跨视角边缘对齐和位姿更新是否稳定，PSNR 作为辅助。
  位姿修正幅度不等于真实位姿误差。
- 使用现有 BA 损失、有效观测数和 pose 指标；必要时低频记录 RGB 与 BA 分别对
  旋转、平移的梯度大小和方向一致性，检查两者是否相互抵消。
- 当前展厅配置为 `use_test_split: false`。实验前确定独立观测或留出视图的评估方案，
  避免仅依据训练视图改善判断 pose 精度。

### 下次先核实的前提与实现注意点

- 核实 LiDAR 几何与相机的整体配准，以及固定参考帧是否可信。如果存在整体外参偏差，
  固定第一帧可能限制纠正空间，需要先评估锚定方式。
- 当前 LiDAR loss 主要约束高斯到局部平面的距离和法向厚度，间接稳定 RGB 的场景依据；
  独立 SfM 轨迹点可学习，BA 提供多视图一致性，不能把两者视为绝对正确的 pose 真值。
- 当前标准 SH 路径已经对视线方向做 `detach()`，展厅配置为 `sh_degree: 0`。
  当前 fisheye + UT + eval3d 通过 eval3d backward 返回相机梯度，不直接照搬笔记中
  二维投影中心和协方差的梯度路径。
- 实现入口主要在 `examples/simple_trainer.py` 的配置、优化器更新、MRNF 调用和
  photometric loss；保留 `examples/pose_refinement.py` 的现有重投影目标。
- 按仓库约定采用最小必要检查，不新增或修改测试；GPU 训练和消融在明确安排后执行。
