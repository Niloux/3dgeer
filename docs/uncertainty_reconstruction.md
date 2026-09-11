# SH0 双不确定性重建

目标是导出稳定的 SH0 PLY：减少运动模糊、局部高光和纹理错位对固定颜色、几何、pose 及增密的影响。pose 与场景继续联合优化，不要求先解决 pose。一次启用完整的观测加权、三维可靠性、增密门控和导出处理。

实现参考 [Hui 等，WACV 2026](https://openaccess.thecvf.com/content/WACV2026/papers/Hui_Joint_Modeling_of_Corruption-Driven_and_Information-Limited_Uncertainty_for_Robust_3D_WACV_2026_paper.pdf) 的双不确定性与软抑制思想，是面向本仓库 SH0、LiDAR 初始化、鱼眼 Eval3D 和 MRNF 的适配，不是论文数值复现。详细研究见[调研记录](research/wacv2026_uncertainty_robust_3dgs.md)。

## 使用

准备好项目 uv 环境后：

```bash
source scripts/activate.sh
uv run python examples/simple_trainer.py \
  --config configs/simple_trainer/exhibition_hall_uncertainty.yaml
```

[完整配置](../configs/simple_trainer/exhibition_hall_uncertainty.yaml) 从现有展厅配置复制，新增 `uncertainty` 参数并使用独立结果目录 `results/0911/展厅_uncertainty`。保留原来的 SH0、PPISP + 局部网格、pose、LiDAR 初始化和其他参数，方便与基线比较。它也沿用 `use_test_split: false`；如需留出视角，需对比较的两组都使用相同划分。

其他配置可以增加 `uncertainty: {enabled: true}`。支持 MRNF + Eval3D、单 GPU、完整图像、`batch_size: 1`、非 packed／非 sparse-grad、训练和导出均为 SH0。与现有 PPISP 或双边网格外观模式组合时，继续遵守原有外观模式互斥规则。PNG 压缩路径不支持此模式，使用 PLY 导出。

## 完整训练闭环

1. **三维污染统计。** 每 `probe_every=50` 步，对外观补偿后、未乘像素可靠性权重的 L1 + SSIM 目标求一次 Gaussian 参数梯度。排除 pose、LiDAR 和外观正则项；不写入优化器 `.grad`。各参数组按当前可见高斯的平均梯度能量归一化，再计算组合幅值的 EMA 均值／二阶矩。`phase_steps=2000` 时重置梯度统计，避免早期梯度尺度长期支配结果。这里采样的是训练过程中访问的视角，不是每次枚举所有视图；探针仍使用当前有效 opacity。
2. **空间聚合。** 每 `refresh_every=500` 步，对高斯中心构建 CPU `cKDTree`，分块查询 8 个近邻，聚合已有梯度证据。使用有限 KNN 邻域，不使用可能连成整个场景的图连通分量。无足够梯度样本的高斯保持中性，单点的强异常不被邻域均值完全抹平。
3. **信息不足统计。** 利用 Eval3D 的 alpha 合成贡献，累计可见次数、最多 `min_views=3` 个不同训练图 ID，以及图像面积归一化的贡献 EMA。结合近邻距离、相对初始化的各向异性增长生成信息不足分数。成熟条件为至少 `min_observations=8` 次可见；多图支持充足时不做这类透明度抑制。不同图 ID 是支持数量的代理，不等价于足够的三角化基线。初始 LiDAR 扁平 surfel 的轴比本身不作为错误。
4. **图像软权重。** 将三维污染与信息不足两个分数作为额外特征通道，与 RGB 在同一次 Eval3D 渲染中合成，再除以 alpha 得到可见贡献加权的分数图。与外观补偿后的局部 RGB／SSIM-CS 残差、每图 32×32 历史误差合并。历史项占局部误差的 25%，当前项占 75%；固定归一化图坐标使数据分辨率切换无需重新分配历史图。所有权重停止梯度，并且能随新观测恢复。
5. **联合优化与增密。** L1 使用像素软权重，SSIM 使用 11×11 窗口权重；分母保留有效像素／窗口数，使整帧降权不会被自身权重归一化抵消。MRNF 在结构误差归一化之后乘同样窗口权重。梯度探针同时通过现有归因缓冲区收集每个高斯的观测置信度，再清空探针归因、恢复训练误差图，避免重复累计。分裂、替换和大尺寸增密只从可见高斯中置信度不低于中位数的候选选取；相同置信度的高斯保持资格。
6. **透明度与状态。** 信息不足高斯使用 `effective_opacity = sigmoid(opacity_logit) * survival`。增密中的低透明度删除使用同一有效值。分裂后父子双方的统计重新收集，confidence／survival 初始化为 1，视图 ID 初始化为 -1；删除时跟随 MRNF 索引更新。高光观测降权不会直接让整个表面进入信息不足抑制。

这是一种拟合可靠性的统计策略，不是精确的高光或运动模糊分类器，也不从模糊观测中显式反演曝光轨迹。它允许不可靠观测被少拟合，以改善静态 SH0 模型。

## 调度与主要参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `warmup_steps` | 1500 | 收集统计，图像权重和 survival 为 1 |
| `ramp_steps` | 1500 | 逐渐增强抑制，不要求 pose 已收敛 |
| `temperature_start` / `temperature_end` | 2.0 / 0.5 | 预热后指数降低温度，随训练收紧软权重 |
| `min_weight` | 0.15 | 有效像素的最低图像监督权重 |
| `min_survival` | 0.1 | survival 乘子的下限；不是最终 alpha 下限 |
| `fusion_weight` | 0.5 | 污染分数相对信息不足分数的比例 |
| `outlier_threshold` | 1.5 | 归一化异常分数的软决策中心 |
| `opacity_strength` | 2.0 | 信息不足指数透明度抑制强度 |

时间参数随现有 `steps_scaler` 缩放。训练预热仍是完整算法内部的调度，不需要单独启动多个训练阶段。当前数字是初始实验设置，尚未由展厅数据验证。

## 保存与验收

- checkpoint 保存原始 Gaussian 参数及不确定性状态，包含最终 survival、每图误差历史、观测图 ID、梯度和支持统计。评估时需要匹配 `uncertainty.enabled` 和原训练图划分，按原有 `--ckpt` 入口加载；该入口仍是评估，不是恢复优化器的续训。
- **PLY 导出有效 opacity 的 logit**，固定 SH0 颜色直接取训练出的原始 Gaussian 颜色。普通 PLY viewer 无需加载置信度图、PPISP 或局部网格，也不会丢掉三维透明度抑制。导出转换在极端值处用 `1e-6` 数值夹紧。
- 普通评估、轨迹和 viewer 渲染使用与导出一致的 survival；训练才渲染两个不确定性附加通道。
- `train.log` 记录平均观测权重、权重小于 0.5 的比例、两类三维分数均值、平均 survival、温度和抑制强度。
- 同时满足日志步与刷新步时，`renders/uncertainty/` 保存横向三联灰度图：左为权重（白色可信），中为污染分数除以 6，右为信息不足分数（后两者白色表示更不可靠）。文件名带 step 和训练图 ID。
- 以导出 PLY 的白斑、重影、浮点、真实亮色纹理保留和表面完整性为主要判断。SH0 只消除显式的方向颜色变化；错误几何和 alpha 混合仍可能造成观察角度变化时的不稳定。

## 计算与验证范围

不修改 CUDA 内核或依赖文件。训练时 RGB 增加两个特征通道，按现有渲染器支持的通道数补齐；每 50 步多一次保留计算图的反向探针，每 500 步多一次 CPU KNN 查询。因此耗时、峰值显存和 CPU 内存会增加，尚无实测速度数据。每高斯状态为 11 个 float32 标量和默认 3 个 int32 视图 ID，约 56 字节／高斯，另有临时渲染和 KNN 缓冲区。

本次没有新增／修改测试，没有启动训练、GPU 执行或 CUDA 构建。静态语法与接口检查已执行；当前项目 `.venv` 缺少 PyTorch，CPU 执行验证未能进行。实际效果与 GPU 渲染兼容性仍需在项目环境完整时验证。
