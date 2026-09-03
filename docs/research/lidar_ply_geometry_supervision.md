# 不生成深度图：直接用 5 cm LiDAR PLY 约束 3D Gaussians

调研日期：2026-09-03

## 结论

当前条件下，建议把 `depth_loss` 保持为 `false`，恢复一条**纯 3D、单向
Gaussian→LiDAR、局部平面、协方差感知**的几何约束。具体做法是：对可靠的
LiDAR KNN-PCA 邻域保存 surfel，在训练时由当前 Gaussian 中心动态查询附近
surfel，以 robust point-to-plane 约束中心，并以 `n^T Sigma n` 限制 Gaussian
穿过表面的法向厚度。

这比 symmetric Chamfer 更适合 5 cm 离散采样：point-to-plane 不惩罚沿切平面的
移动，因此 densification 出来的 Gaussian 可以填补近景的屏幕空间空隙，而不会
被拉回某个离散 LiDAR 点。LI-GS 也用可见 Gaussian 到附近 LiDAR 平面/GMM 的
加权 point-to-plane 距离，并另约束形状与法向；其消融说明“只有 LiDAR/GMM
初始化”不足以防止后续 photometric optimization 破坏几何。
[LI-GS Eq. 9--13 与消融](https://arxiv.org/html/2409.12899#S4.SS3.SSS1)

仓库历史 commit `65f5148` 已经有接近这一目标的最小
`LidarSurfelField`，commit `d7eea6b` 才将其替换为深度图。最小可落地路径不是
重新设计一套 SDF，而是恢复该无状态模块，并修正它的空间索引、匹配门控和
法向厚度目标。本文只给方案，不修改训练代码或测试。

```bash
git show 65f5148:examples/lidar_geometry.py
git show d7eea6b -- examples/lidar_geometry.py examples/simple_trainer.py
```

另一项 2026 年的直接 LiDAR-supervised surface-aligned 3DGS 工作也明确选择用
LiDAR/SfM 的 position 和 normal 联合约束 Gaussian position 与 anisotropic
shape，而不先生成 depth map；不过当前可访问的出版页没有足够实现细节，因此
本文不从它推断具体公式。
[Direct LiDAR-Supervised Surface-Aligned 3D Gaussian Splatting](https://doi.org/10.1016/j.displa.2026.103349)

## 先澄清“近处更稀”：投影密度不等于米制密度

Park 的 `lidar.ply` 有 1,488,579 点。对随机 50,000 个点做本地 KNN 审计，距离
统计为：

| 邻居距离 | p50 | p90 |
| --- | ---: | ---: |
| `d2` | 3.38 cm | 5.57 cm |
| `d24` | 13.04 cm | 21.52 cm |

再按“到最近相机中心的距离”分桶，`d24` 的 p50 为：

| 最近相机距离 | 0--5 m | 5--10 m | 10--20 m | 20--40 m |
| --- | ---: | ---: | ---: | ---: |
| `d24` p50 | 12.48 cm | 13.18 cm | 12.44 cm | 16.40 cm |

因此在 0--20 m 内，近处并没有表现出更差的**米制**邻域支持；20--40 m 反而
略稀。近处“看起来更稀”主要是固定 5 cm 间距的透视结果：焦距为 `f`、深度为
`z` 时，5 cm 横向间隔约投影成 `Delta u ~= f * 0.05 / z` 个像素，`z` 越小，
屏幕上的点间空隙越大。

这一区分直接决定设计：

- 不应按“离全局原点/最近相机的距离”给 3D loss 加权。融合后的全局 PLY 没有
  唯一传感器 range，而且本地数据也不支持“近处米制更稀”这个假设。
- 局部平面 loss 负责“新 Gaussian 仍在真实表面上”；屏幕空间覆盖仍应由 RGB
  误差和现有 densification 负责。point-to-plane 在切向不施力，正好允许新增
  Gaussian 填 5 cm 点之间的近景投影空隙。
- 5 cm 只是下采样格距，不等于 5 cm 几何带宽。当前 `k=24` 的平面邻域中位
  半径已经约 13 cm；若要保留路沿、细杆等 5--10 cm 结构，还要避免大邻域跨面。
  PCL 的 PCA normal 文档也明确指出，邻域过大会跨越相邻表面，模糊边缘和细节。
  [PCL normal estimation](https://pointclouds.org/documentation/tutorials/normal_estimation.html)

## 方案比较

| 方案 | 能解决什么 | 当前主要问题 | 本项目结论 |
| --- | --- | --- | --- |
| Gaussian 中心→LiDAR 点，一侧 NN | 抑制靠近 PLY 的 floaters；实现简单 | 惩罚切向移动，过拟合离散 5 cm 采样；空洞处会吸向错误表面 | 只作 sanity baseline |
| LiDAR 点→Gaussian 中心，一侧 NN | 保证 LiDAR coverage | 不惩罚额外 floaters；高密度区域主导；初始化时几乎冗余 | v1 不用；出现孔洞再低权重加入 |
| symmetric Chamfer | 同时约束 accuracy 与 completeness | 强迫 Gaussian 复刻 LiDAR 的采样分布；噪点、动态点和不完整 coverage 都变成硬目标 | 不推荐作训练主 loss |
| **Gaussian→局部 LiDAR 平面** | 约束法向偏移，同时允许切向重采样 | 需要可靠 normal、局部支持门控和动态 NN | **推荐** |
| Gaussian likelihood / Mahalanobis | 把点解释为 Gaussian 分布样本 | 单独 Mahalanobis 可通过放大协方差降 loss；加 `log det` 后又把二维表面当三维体分布 | 不推荐 |
| SDF/UDF | 连续表面、可查询梯度 | 需额外 sparse grid/MLP 与正则；只有 surface samples 时符号和自由空间不可靠 | 后续重方案 |
| occupancy | 表达 occupied/free/unknown | PLY 只给 hit，没有 free/negative label，会有“全占据”退化 | 当前不可辨识 |
| LiDAR ray/free-space | 能直接惩罚 return 前的 floaters 和错误遮挡 | 必须保留每个 return 的 scan origin/pose；融合 PLY 已丢失 | 有 raw sweep 后再做 |
| 独立 normal consistency | 约束朝向 | PCA normal 有正负歧义；最短轴选择在 scale 次序交换时不连续 | 不单独启用，先用 `n^T Sigma n` |

PyTorch3D 的官方 Chamfer 实现清楚区分 `single_directional` 与双向距离，并允许
normal loss 使用 `abs_cosine` 消除法向正负号；这正是上表所说的方向语义。
[PyTorch3D `chamfer_distance`](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/loss/chamfer.py)
Density-aware Chamfer 原论文则指出普通 Chamfer 对局部密度失配不敏感且无界
距离容易受 outlier 支配；这说明不能把“一个平均 Chamfer 变小”直接解释为采样
分布或局部细节正确。[Density-aware Chamfer, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/f3bd5ad57c8389a8a1a541a76be463bf-Abstract.html)

## 推荐的最小损失

### 1. 一次性构建可靠 LiDAR surfels

沿用当前 [KNN-PCA 初始化](../../examples/gaussian_models.py#L70-L177)，但不要在
初始化结束后丢掉中间量。对每个可靠点保存：

```text
p_j       KNN 点的质心（不是带噪的原始中心点）
n_j       协方差最小特征值对应的单位特征向量
r_j       第 k 个邻居的支持半径
sigma_j   局部切向尺度，建议 sqrt(s_t1 * s_t2)
tau_j     初始化时的法向 scale
c_j       planarity / curvature / 平面残差得到的 [0,1] 置信度
```

PCA 平面应通过邻域质心；最小特征向量是 normal，曲率常用
`lambda_0 / (lambda_0+lambda_1+lambda_2)`。PCA 无法自行确定 normal 的正负号，
所以后续所有法向比较都必须符号无关。
[PCL 的公式、质心、曲率与符号歧义](https://pointclouds.org/documentation/tutorials/normal_estimation.html)

当前 [Park 配置](../../configs/simple_trainer/park.yaml#L29-L44) 的门槛
`planarity >= 0.30`、`curvature <= 0.10` 可先原样复用。`k=24` 是稳定
baseline；细节场景再消融 `k=12`，或从 `{8,12,24}` 中选择“最小且已经稳定为
平面”的邻域。不要一开始同时更换 normal estimator 和训练 loss。

### 2. 动态关联，而不是固定 anchor

对本次选中的前景 Gaussian `i`，用 `mu_i.detach()` 查询最多 `K=4` 个可靠
surfel。LI-GS 也是对每个可见 Gaussian 使用 4 个邻近 GMM component，并按空间
距离加权；其论文参数 `K=4` 可作起点，但其它量纲参数不能直接搬到 Park。
[LI-GS Eq. 9--10 与参数表](https://arxiv.org/html/2409.12899#S4.SS3.SSS1)

候选必须满足：

```text
||mu_i - p_j|| <= R_j
R_j = clamp(1.5 * r_j, 2h, 0.30 m),  h = 0.05 m
```

所有米制阈值需随 parser 的 similarity scale 一次性转换到训练坐标。多候选只
保留与最近可靠 normal 满足 `|n_j^T n_ref| >= cos(30 deg)` 的同一法向簇，避免
在墙角、路沿和前后表面之间平均。对应关系和权重均 detach；梯度只通过选中的
平面残差传回 Gaussian 参数。

令

```text
a_ij = c_j * exp(-||mu_i-p_j||^2 / (2 r_j^2))
a_bar_ij = a_ij / sum_j(a_ij)
q_i = max_j(a_ij)
```

`a_bar` 只负责在同一 Gaussian 的多个平面之间混合，外层
`weighted_mean_i` 用 detached 的 `q_i` 表示本次关联的绝对可靠性；否则只有一个
低置信候选时，归一化会错误地把它恢复成权重 1。

若没有通过支持门控的候选，该 Gaussian 本次不产生 geometry gradient，不能把
它硬拉到很远的“最近点”。Splat-LOAM 的正式论文/实现同样采用 PCA 平面、
correspondence gating 和 robust point-to-plane，而不是无门控全局吸附。
[Splat-LOAM, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Giacomini_Splat-LOAM_Gaussian_Splatting_LiDAR_Odometry_and_Mapping_ICCV_2025_paper.html)

### 3. 中心和协方差一起约束

对 Gaussian

```text
Sigma_i = R_i diag(s_i^2) R_i^T
```

及候选平面 `(p_j,n_j)`，定义

```text
e_ij = n_j^T (mu_i - p_j)            # 中心到平面的有符号距离
t_ij = sqrt(n_j^T Sigma_i n_j)       # Gaussian 沿平面法向的标准差
```

`t_ij` 同时对 rotation 和各轴 scale 可导，不需要判断“当前哪根轴最短”，也不
受 `n` 与 `-n` 影响。它还是下面恒等式中的精确法向二阶矩：

```text
X ~ N(mu_i, Sigma_i)
E[(n_j^T(X-p_j))^2] = e_ij^2 + t_ij^2
```

历史 `65f5148` 正是优化

```text
rho(sqrt(e_ij^2 + t_ij^2) / sigma_j)
```

其中 `rho` 为 SmoothL1，并以 surfel confidence 加权。它是很好的复现 baseline，
但它在中心已正确时仍持续把 `t` 推向 0，最终法向厚度由 scale clamp 决定。

建议正式 v1 做一个很小的安全修改：把中心贴面与“超过初始化厚度”的部分拆开，
只把厚度当上界约束：

```text
L_center(i,j) = SmoothL1(e_ij / sigma_j, 0; beta=1)
L_thick(i,j)  = SmoothL1(relu(t_ij - tau_j) / sigma_j, 0; beta=1)

ell_i = sum_j a_bar_ij * (L_center(i,j) + beta_t * L_thick(i,j))
L_geo = sum_i q_i * ell_i / (sum_i q_i + eps)
```

初值建议 `beta_t=0.25`，`tau_j` 直接取当前 KNN-PCA 初始化得到的 normal scale
（当前配置即 `0.25 * base`）。`sigma_j` 建议使用历史实现的切向几何均值，并
clamp 到 `[0.5h, 2h]`；这样 loss 无量纲，稀疏邻域容忍度略大，但不会因极大
support radius 失去约束。

SmoothL1/Huber 是为了降低错误 correspondence 和离群点的影响。Open3D 的官方
robust point-to-plane 文档给出了 `sum rho((p-Tq).n)` 及其 IRLS 权重解释。
[Open3D robust kernels](https://open3d.org/docs/latest/tutorial/pipelines/robust_kernels.html)

不要把 opacity 乘进可学习权重，否则 Gaussian 可以通过降低 opacity 逃避几何
loss。可以用 detached 的 visible/opacity mask 省算力，但分母只使用 detached
置信度。sky Gaussians 必须完全排除。

### 4. 为什么不再单加 normal loss

直接法向项可写成 `1-|n_j^T u_min,i|`，符号无关；LI-GS 的确同时用了中心、
control points 和 normal loss。[LI-GS Eq. 11--13](https://arxiv.org/html/2409.12899#S4.SS3.SSS1)
2DGS 则把 rendered normal 与 rendered median-depth 的表面 normal 对齐，这仍然
依赖渲染深度，并不是本任务所需的纯 3D 约束。
[2DGS 论文](https://www.cvlibs.net/publications/Huang2024SIGGRAPH.pdf)

本项目先不加独立 normal loss：`n^T Sigma n` 已经平滑地推动薄轴朝向 LiDAR
normal，并避免 `argmin(scale)` 在轴次序交换时跳变。只有消融显示 `t` 已小但
朝向仍不稳定时，再加很小的 `1-|dot|`；不要同时上线两个含义重叠的大权重项。

## 必须修正历史空间索引

`65f5148` 的主要问题不在 loss，而在查询结构：

1. 它把可靠点 `d24` support radius 的**全局 median**当作 index voxel size；
   Park 上约为 13 cm，而输入本身是名义 5 cm voxel 点云。
2. 每个 index voxel 只保留 confidence 最高的一个 surfel，造成第二次、且更粗的
   下采样，可能消掉 5--13 cm 结构。
3. 每次只查固定 `3x3x3` cell。这不是有物理半径保证的查询；可靠 surfel 稀疏、
   邻近 cell 为空时会直接无匹配。

建议改为：

- index cell 固定为源数据的 `h=5 cm`（经 scene transform 缩放），不要由 `d24`
  的全局 median 决定；
- 每个 cell 使用 CSR/list 保留全部可靠 surfel；若显存必须限流，在候选阶段按
  confidence/距离截到 32--64 个，而不是建索引时永久只留一个；
- 先查半径 1 cell，再只对未得到足够候选的 query 扩到 2、4、最多 6 cell，且
  最终仍执行上面的 `R_j` 米制门控；分块 query，避免一次物化巨大候选张量；
- 每个 geometry update 都根据当前 `mu.detach()` 重新关联。这样 split/clone/
  prune 后无需继承 anchor ID，也不会把子 Gaussian 永久锁在旧平面。

`d24` 的 p90 为 21.52 cm，因此“5 cm index + 只查 3x3x3”也不够；另一方面，
直接构造完整 `13^3` 邻域又浪费显存。逐级扩圈且只处理 unmatched queries 是兼顾
召回和成本的最小实现。

## 密度、遮挡、动态物体与噪声

### 密度不均

5 cm voxel downsample 已经抑制了局部过密，Park 的距离分桶也不支持默认做
near/far reweight。v1 使用局部 `sigma_j` 归一化和 `c_j` 可靠性即可。如果日志
仍显示少数高密度区域主导，再按 0.5 m 粗 occupied blocks 均匀抽样，而不是按
点数或相机 range 抽样；也可将每点代表面积近似为 `r_j^2`，但权重必须 clamp
（例如相对 median 的 `0.25--4x`），避免把稀疏 outlier 放大。

### 遮挡和表面不完整

纯 3D point-to-plane 不需要相机 z-buffer，因此没有“把累计全局点云投到每张图
时多个时刻/多个表面争一个像素”的预处理问题。LidaRF 也报告多帧累计 LiDAR
投影会产生多层 surface、ghost depth 和边界歧义。
[LidaRF §4.3](https://arxiv.org/html/2405.00900#S4.SS3)

代价是它不表达前后可见性和自由空间。邻近的两面可能错配，所以必须使用小
index、support radius、同法向簇和 robust loss；LiDAR 无覆盖处应为“无约束”，
不能用远距离 NN 猜一个表面。

### 动态物体

Huber 只能抑制孤立 ghost，不能识别一辆完整、局部很平的车；这种动态物体反而
会形成高置信平面。LI-GS 在形成全局地图前使用 M-detector 删除动态点，并用 HBA
改善地图一致性，说明动态清理属于 scan-level preprocessing，而不是靠最终
point-to-plane loss 自动解决。
[LI-GS preprocessing](https://arxiv.org/html/2409.12899#S4.SS1)

若还保留逐帧 scans 和 poses，可用
[ERASOR](https://github.com/LimHyungTae/ERASOR) 或
[Removert](https://github.com/gisbi-kim/removert) 生成静态 PLY；两者的官方接口都
依赖逐帧点云与位姿。若只剩融合 PLY，可靠选择只有语义/人工 mask 或把可疑区域
`c_j=0`，不能仅凭局部几何保证区分动态与静态实体。

### 噪声和边界

- 平面 origin 用邻域质心；用 planarity、curvature、支持半径和法向残差构造
  soft confidence。
- 对 `r_j > 0.30 m`、非平面、线状、混合 surface 邻域直接不监督。
- 先 soft downweight + SmoothL1，不要激进 radius/statistical outlier deletion；
  官方 Open3D 文档展示的 outlier removal 本质仍基于邻居数量/平均距离，真实稀疏
  区也可能被删。[Open3D outlier removal](https://www.open3d.org/docs/release/tutorial/geometry/pointcloud_outlier_removal.html)
- 5--10 cm 边缘细节若重要，优先做多尺度 normal 稳定性检查，而不是把 index
  voxel 放大到 `d24` 来获得更多候选。

## 为什么当前不做 SDF、occupancy 和 ray loss

SHINE-Mapping 证明可以用 sparse octree features + shallow MLP 从 LiDAR 学连续
SDF，但其官方实现还要在 SDF BCE/L1/L2、ray rendering 和 Eikonal 等训练项之间
选择；这是一套额外表示和优化过程，不是“直接复用 PLY”的低成本 regularizer。
[SHINE-Mapping 论文](https://arxiv.org/abs/2210.02299)，[官方实现与 loss 选项](https://github.com/qixuema/SHINE_mapping)
IGR 可以只从 unoriented point samples 学一个 plausible implicit surface，但同样
需要额外网络与 Eikonal/normal 约束，且开放、稀疏大场景中的补面是模型先验，
不是观测到的 free space。[IGR, ICML 2020](https://proceedings.mlr.press/v119/gropp20a.html)

occupancy/TSDF 的关键监督不是 endpoint 本身，而是 ray 上的 hit 与 miss。
OctoMap 官方 API 的 `insertRay(origin,end)` 会把 endpoint 更新为 occupied，并把
之前的体素更新为 free；没有 origin 就缺了后一半信息。
[OctoMap ray integration](https://octomap.github.io/octomap/doc/classoctomap_1_1OccupancyOcTreeBase.html)
Voxblox 也从 posed sensor data 沿测量 ray 融合 TSDF。
[Voxblox 原论文](https://arxiv.org/abs/1611.03631)

真正的 LiDAR line-of-sight loss 可对 return 前的密度/权重施加 empty-space
惩罚；Urban Radiance Fields 的 LiDAR 样本显式包含 `(origin, direction, range)`，
并分别定义 near-surface 和 empty-space 项。
[Urban Radiance Fields, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Rematas_Urban_Radiance_Fields_CVPR_2022_paper.pdf)
融合 `lidar.ply` 没有 point→scan 对应和传感器 origin，不能拿“最近相机中心”
伪造 LiDAR ray：传感器基线、时间、遮挡都可能不同。若未来保留 raw sweep，ray
loss 是很有价值的第二阶段，而且仍不必生成相机 depth PNG；但不属于当前 v1。

## 训练和性能起点

下表是结合 Park 统计与当前配置给出的**工程起点**，不是论文报告的通用最优值：

| 项目 | 建议起点 | 说明 |
| --- | ---: | --- |
| `geometry_lambda` | `0.01` | sweep `0.003 / 0.01 / 0.03` |
| weight warmup | `0 -> lambda`，500 steps | 降低外观初始瞬态冲击 |
| geometry cadence | 每 4 steps | 无需每张图都做完整 NN |
| sampled foreground GS | 32,768/update | 先于历史 100,000 降成本 |
| local planes per GS | 最多 4 | 与 LI-GS 的 `K=4` 起点一致 |
| PCA `k` | 24 | 复用当前配置；细节消融 12/多尺度 |
| index voxel | 0.05 m | 保留输入分辨率，不用全局 `d24` median |
| match radius | `clamp(1.5r, 0.10, 0.30)m` | 覆盖 Park 的支持半径分布，又有硬上限 |
| normal cluster | 30 deg，使用 `abs(dot)` | 防止跨相邻面混合 |
| `beta_t` | 0.25 | 厚度为辅助约束 |
| SmoothL1 `beta` | 1（归一化单位） | 转折点由 `sigma_j` 给出 |

更可靠的 `geometry_lambda` 选择方法是偶尔记录两个 loss 对 `means` 的梯度范数，
让 `lambda * ||grad L_geo||` 在早期大致为 RGB loss 对 `means` 梯度的 5--20%，而不是盲目
照搬 LI-GS 的 `lambda_GMM=1` 或历史实现的 `0.10`。二者的 reduction、表示和
尺度都不同。

Park 约 149 万 surfels。仅 `xyz + normal + confidence + support + sigma + tau`
的 float32 原始数组约 60 MB；加 hash keys/CSR 后仍应在百 MB 量级，但 radius
候选临时张量可能远大于索引本身。因此应以 4k--8k query chunks 搜索、总计最多
32k Gaussian，并对远圈只处理尚未匹配的 query。LI-GS 也报告 nearest-neighbor
GMM normalization 令训练比 2DGS 略慢，所以必须实测 step time/VRAM。
[LI-GS 训练开销说明](https://arxiv.org/html/2409.12899#S5.SS2)

当前 trainer 的 LiDAR 初始化会把 PLY 变换到训练坐标后做 KNN-PCA，并把
`means/quats/scales` 作为自由参数参与后续 densification。
[初始化调用](../../examples/simple_trainer.py#L703-L788)
恢复的 loss 只作用于前景 `self.splats`；固定 sky splats 不参与。直接 3D loss
也不会经过 rasterizer，因此不会给 camera pose/intrinsics 提供直接梯度：它把
Gaussian 固定在 LiDAR 世界坐标中，pose/calibration 仍由 RGB loss 优化。这是相对
深度图监督必须接受的明确取舍。

## 分阶段验证

建议只做四个有信息量的消融，不要一次引入 SDF 或 ray infrastructure：

1. `G0`：当前 `depth_loss=false`，没有 geometry loss。
2. `G1`：恢复历史 expected-plane loss，但使用修正后的 5 cm index、支持门控和
   32k/4-step sampling，用来判断历史 loss 本身的收益。
3. `G2`：改成推荐的 center + excess-thickness loss，验证是否减少 scale collapse
   或 needle/纸片异常。
4. 只有出现明确 LiDAR coverage 孔洞时，才试低权重、按粗 occupied block 均匀
   采样的 `LiDAR→Gaussian` 项；不要直接上 symmetric Chamfer。

每个实验至少记录：

- RGB：test PSNR/SSIM/LPIPS；
- geometry：有效匹配率、`|e|` p50/p90、`t` p90、LiDAR→Gaussian coverage；
- 分桶：按 `d24` quartile 和最近相机距离报告匹配率/残差，避免平均数掩盖稀疏区；
- 结构：Gaussian 数量、长期 unsupported 且低 opacity 的数量；
- 性能：平均 step time、geometry update time、峰值 VRAM。

训练和评估若使用同一 PLY，只能证明“贴合 prior”，不能证明真实几何泛化。最好
用 withheld LiDAR sweeps 或独立清理后的点云评估 accuracy/completeness。LI-GS
同样同时报告 accuracy、completeness、Chamfer-L1、precision/recall/F-score，而
不是只看渲染指标。[LI-GS evaluation](https://arxiv.org/html/2409.12899#S5.SS1)

在确认 match statistics 可靠之前，不建议把几何距离直接接入 prune。后续若要
做，可借鉴 LI-GS 将 surface distance 放入 grow/prune 的做法，但在本项目必须
要求“多次可见 + 长期 unsupported/high residual + 低 opacity”同时成立，不能因
PLY 空洞单独删点。[LI-GS Eq. 17--18](https://arxiv.org/html/2409.12899#S4.SS3.SSS2)

## 最终决策边界

- **现在就做**：修正版 `LidarSurfelField`、动态支持门控 point-to-plane、
  covariance thickness、robust loss、细粒度 hash、有限采样。
- **明确不做**：预生成深度/法向图、默认 symmetric Chamfer、全局 range 权重、
  从相机中心伪造 LiDAR ray、无负样本 occupancy。
- **有证据再做**：低权重 LiDAR→Gaussian coverage、独立 normal loss、
  geometry-aware pruning、多尺度 PCA。
- **只有保留 raw scans/origins 才做**：free-space/ray loss、TSDF/occupancy；若
  局部平面在复杂结构上确实失败，才考虑 SDF/NKSR 类二阶段 surface field。
