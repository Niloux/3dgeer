# 独立 SfM 特征轨迹约束

`examples/simple_trainer.py` 的 `pose_opt` 统一使用独立 SfM 特征轨迹约束。
开启后，在 pose 优化阶段同时优化相机和共享三维特征点。
相机同时接收图像损失和重投影损失的梯度；独立的三维特征点只接收
重投影梯度，不与高斯位置绑定，也不参与 MRNF 的增密、删除和迁移。
该设计借鉴 [GloSplat §3.3、附录 E](https://arxiv.org/html/2603.04847v1#S3.SS3)，
沿用现有 SfM 前端和 Gaussian 初始化流程。

## 运行展厅预设

```bash
source scripts/activate.sh
uv run python examples/simple_trainer.py \
  --config configs/simple_trainer/exhibition_hall_pose_tracks.yaml
```

这是基于已完成的 `results/0911/展厅base/cfg.yml` 整理的完整参数快照。
采用展厅对照后选择的轨迹权重 `0.01`，输出目录为
`results/0912/展厅_pose_tracks`。SH0、LiDAR、MRNF、曝光校正、
pose 学习率、500 步启动、20000 步冻结和 15000 总步数均沿用原值。
配置头部记录原始快照的 SHA-256。

```yaml
pose_opt: true
pose_track_lambda: 0.01
pose_track_lr: 0.0001
pose_track_min_views: 3
pose_track_max_reprojection_error: 2.0
pose_track_huber_delta: 1.0
```

当前支持单 GPU、`batch_size: 1`、`patch_size: null`、
`keep_distortion: true`，相机为 pinhole 或 OpenCV fisheye。
通用配置的 `pose_opt` 默认关闭，展厅轨迹预设开启。
启用 pose 优化后，轨迹权重和轨迹点学习率必须为正；若无合格轨迹，
初始化会报错。`pose_opt: false` 同时关闭相机和轨迹点优化，不加载轨迹观测。

独立的实验开关 `pose_track_enabled` 已删除。旧 YAML 快照需要移除该字段；
`pose_opt: true` 的训练始终使用新方案，不再提供无轨迹约束的 pose 训练分支。
`pose_opt_lr`、平移/旋转学习率、参考相机和弱位姿先验等参数继续沿用，
保持已验证的优化设置；替换监督方案不需要更换相机的位姿参数化。

## 观测与损失

二维目标直接取 `images.bin` 中的实际特征观测，并通过 `points3D.bin`
的 track 关联同一个三维点。训练图像排序、缺失图像、frame 范围筛选和
训练/验证划分均按 parser 和 trainset 的实际索引处理；验证图像的二维观测
不进入此训练损失。三维点初始化沿用现有 SfM 及其场景坐标变换。

初始化时，在初始训练分辨率下过滤非有限值、相机后方点、超过 2 px 的
重投影残差，以及 FOV、数据有效掩码和已加载天空掩码之外的观测。
过滤后至少在 3 张训练图像中可见的点才参与优化。筛选只做一次；
后续误差变大不会触发删除。没有合格观测的单张图像贡献零损失，
启动日志会列出这些图像。

每步使用当前训练图像的全部保留观测，共享三维特征点将各视图联系起来。
设像素残差长度为 `r`，Huber 阈值为 `d`：

```text
h(r) = 0.5 * r²                   (r <= d)
     = d * (r - 0.5 * d)          (r > d)
L_track = mean(h(r))
L_total = 图像及高斯损失 + 弱位姿先验 + pose_track_lambda * L_track
```

投影使用原图相机模型、畸变和当前训练内参；二维观测按实际图像尺寸同步缩放。
COLMAP 观测沿用像素角点原点约定，不额外加减半像素。
分辨率切换会改变像素残差的单位尺度，初始筛选集合则保持固定。

这里使用逐图 **mean**，所以本实现的权重不能与论文求和形式的 `1e-4`
直接比较。展厅的 `0.01` 与 `0.02` 对照后采用 `0.01`，不代表其他场景的最优值。
三维点使用独立 SparseAdam；`pose_track_lr` 是训练世界坐标中的步长，
不乘高斯的场景尺度。其学习率按原 pose 的指数衰减规则变化，
相机和轨迹点共用优化器更新及学习率调度入口，只有 pose 活跃且本步训练
更新有效时才推进；warmup、冻结和跳过非有限 loss 时两者都不更新。
原有参考相机和 pose 先验继续生效；轨迹约束本身不固定绝对尺度。

## 日志与保存

`train.log` 增加 `pose_tracks_init` 事件，记录保留点数、观测数、
没有轨迹的图像及初始像素残差。周期性训练日志包含：

- `pose_track_active`、`pose_track_point_lr`；
- `pose_track_loss` 和实际加入总损失的 `pose_track_weighted_loss`；
- `pose_track_observations`，当前图像参与的观测数；
- `pose_track_reprojection_mean_px` / `median_px` / `p90_px`；
- `pose_track_behind_camera_fraction`，用于发现轨迹几何退化。

残差是当前图像上、相机和特征点共同拟合后的训练量，不能充当独立验证。
warmup 和冻结阶段只在日志步计算该量，不反向传播。

checkpoint 的 `pose_tracks` 项保存三维点、原始 COLMAP 点 ID、
固定二维观测、图像索引分段、相机畸变和图像名称。
它们不写入 PLY。`--ckpt` 仍然是已有的仅评估入口，渲染使用保存的 pose 和
高斯，不重新筛选或拟合轨迹；当前 trainer 不提供完整训练断点续跑。

历史结果可继续用于 pose 关闭、旧 pose、新 pose 方案的对照。
当前代码不再重新训练旧分支。调整轨迹权重时，应检查独立匹配的重投影、
同位置文字裁剪和多个共视角，并同时比较全图指标。
只有训练 track 残差降低，不能证明文字改善或新视角泛化提高。
