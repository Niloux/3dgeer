# Energy-GS 思想整理：训练中联合优化 Camera Pose 与 3D Gaussian Splatting

> 论文：**Energy-GS: Image Energy-guided Pose Alignment Gaussian Splatting with redesigned pose gradient flow**  
> CVPR 2026，Yu Gao et al.  
> 本文档基于论文主文与 Supplementary Material，对其核心思想、优化机制与工程启示进行整理。

---

## 1. 这篇论文想解决什么问题？

Energy-GS 解决的是：

> **在 3D Gaussian Splatting（3DGS）训练过程中，仅依赖 RGB 图像，同时优化场景表示和相机 Pose。**

最直接的做法是把 camera pose 设为可学习参数，然后让 photometric loss 反向传播：

\[
I_{\text{render}} = \mathcal{R}(G,T)
\]

\[
L = L_{\text{photo}}(I_{\text{render}}, I_{\text{GT}})
\]

其中：

- \(G\)：Gaussian primitives
- \(T\)：camera pose
- \(\mathcal{R}\)：3DGS rasterizer

理论上 RGB reconstruction loss 可以同时更新 Gaussian 和 Pose。

但作者发现，**vanilla 3DGS 的 joint optimization 非常容易不稳定或落入错误局部最优**。

Energy-GS 的核心不是增加额外几何先验，而是重新设计优化过程，使 **RGB supervision 本身更适合优化 Pose**。

---

# 2. 核心观察：3DGS 会“吸收”Pose 误差

Pose 不准确时，rendered image 与 GT 会错位。

理论上优化器应该：

```text
RGB mismatch
    ↓
pose gradient
    ↓
camera 向正确位置移动
```

但在 3DGS 中，同时可学习的变量很多：

```text
camera pose

Gaussian xyz
Gaussian scale
Gaussian rotation
Gaussian opacity
Gaussian color / SH
Gaussian clone / split / prune
```

于是 optimizer 可能走另一条更容易的路径：

```text
RGB mismatch
    ↓
Gaussian 自己移动 / 变形 / 分裂 / 改颜色
    ↓
当前视角 reconstruction loss 下降
    ↓
Pose 仍然是错的
```

这导致一个非常重要的问题：

> **高质量的局部 rendering，并不能证明 camera pose 已经对齐。**

Supplementary Material 中的 1DGS 实验专门验证了这一点：局部信号可以被 Gaussian 很好地重建，但其空间位置仍然可能没有真正对齐。

---

# 3. Energy-GS 的总体思想

Energy-GS 可以概括成两条主线：

## 3.1 限制 Gaussian 在 Pose 对齐阶段“作弊”

在 Pose 尚未稳定时：

- 固定 Gaussian position
- 固定初始 Gaussian 数量
- 暂停 densification
- 避免动态 primitive 集合破坏 Pose gradient

目标是：

> **让 reconstruction error 更多地通过 Pose 被解释，而不是由 Gaussian geometry 自己吸收。**

---

## 3.2 把 RGB supervision 从“难”变成“由易到难”

普通 3DGS：

```text
iteration 0
    ↓
直接使用 full-detail RGB
    ↓
复杂纹理 / 高频细节 / 局部结构全部参与 pose optimization
```

Energy-GS：

```text
coarse / low-energy image
        ↓
逐渐加入更多图像结构
        ↓
full-detail RGB
```

目标类似 BARF 的 coarse-to-fine pose alignment：

> **先对齐大的结构和整体位置，再逐步学习细节。**

由于 3DGS 没有 NeRF 那样可控的 positional encoding frequency，Energy-GS 转而对 **GT 图像自身做 energy decomposition**。

---

# 4. Pose 如何参数化？

论文把 camera pose 表示成 Lie algebra 中的可学习变量：

\[
\xi_i \in \mathfrak{se}(3)
\]

再通过指数映射得到：

\[
T_i \in SE(3)
\]

工程上可以理解为：

```python
pose_delta = nn.Parameter(torch.zeros(N, 6))

T_refined[i] = SE3.exp(pose_delta[i]) @ T_init[i]
```

这种参数化有几个优点：

- rotation / translation 统一表示
- 更新天然处在 SE(3) 流形附近
- 适合 gradient-based optimization
- 比直接裸优化 rotation matrix 更合理

---

## 4.1 固定第一台相机

Energy-GS 在 joint optimization 中固定第一台 camera：

```text
camera 0:
    fixed

camera 1 ... N:
    learnable
```

原因是多视图 reconstruction 存在 global gauge freedom。

如果所有 camera 都自由变化，那么整个场景和相机系统可以整体发生刚体变换，而 loss 不一定改变。

固定第一台相机相当于给整个系统设置坐标系 anchor。

---

# 5. 作者分析了 Pose gradient 到底从哪里来

Supplementary Material 将 camera pose 的反向传播路径分成三类，按影响从强到弱排列。

---

## 5.1 Path 1：通过 Gaussian alpha

第一条，也是最重要的一条：

\[
L
\rightarrow
\alpha_i
\rightarrow
T
\rightarrow
se(3)
\]

其中 \(\alpha_i\) 是某个 Gaussian 对 pixel 的 alpha contribution。

实际上又可以拆成：

\[
L
\rightarrow
\alpha_i
\rightarrow
\mu
\rightarrow
T
\rightarrow
se(3)
\]

以及：

\[
L
\rightarrow
\alpha_i
\rightarrow
\Sigma'
\rightarrow
T
\rightarrow
se(3)
\]

其中：

- \(\mu\)：Gaussian 投影到屏幕后的 2D center
- \(\Sigma'\)：Gaussian 的 2D projected covariance

---

# 6. Pose 如何通过 projected center 获得梯度？

世界坐标中的 Gaussian center：

\[
\mu_g
\]

经过 camera transformation：

\[
X_c = T\mu_g
\]

再经过 perspective projection：

\[
\mu = \pi(T\mu_g)
\]

因此：

\[
\frac{\partial \mu}{\partial se(3)}
=
J_\pi J_T
\]

其中 perspective projection Jacobian：

\[
J_\pi =
\begin{bmatrix}
1/Z & 0 & -X/Z^2 \\
0 & 1/Z & -Y/Z^2
\end{bmatrix}
\]

而 world-to-camera transform 对 \(se(3)\) 的 Jacobian 可写成：

\[
J_T =
\begin{bmatrix}
I & -[T\mu_g]_\wedge
\end{bmatrix}
\]

因此 Pose 变化会导致：

```text
Pose
 ↓
Gaussian projected center 变化
 ↓
pixel alpha contribution 变化
 ↓
rendered RGB 变化
 ↓
photometric loss
```

这就是 RGB loss 可以反向优化 camera pose 的主要机制之一。

---

# 7. Pose 还会改变 Gaussian 的 projected covariance

3D Gaussian covariance 为：

\[
\Sigma
\]

投影到图像后：

\[
\Sigma'
=
JW\Sigma W^T J^T
\]

其中：

- \(W\)：camera rotation
- \(J\)：projection Jacobian

因此 Pose 尤其是 camera rotation 变化时：

```text
rotation
   ↓
projected Gaussian ellipse shape
   ↓
alpha distribution
   ↓
rendering
   ↓
loss
```

所以第一条完整的 gradient path 是：

\[
L
\rightarrow
\alpha
\rightarrow
\begin{cases}
\mu \\
\Sigma'
\end{cases}
\rightarrow
T
\rightarrow
se(3)
\]

---

# 8. Path 2：通过 accumulated transmittance / visibility

3DGS alpha compositing 中：

\[
acc_i =
\prod_{j<i}(1-\alpha_j)
\]

因此某个前景 Gaussian 的 alpha 发生变化，会影响后面 Gaussian 的 visibility。

产生：

\[
L
\rightarrow
acc_i
\rightarrow
\alpha_j
\rightarrow
T
\rightarrow
se(3)
\]

可以理解为：

> **遮挡关系和前后可见性本身也可以提供 Pose gradient。**

因此 Pose optimization 并不只是依赖 projected center alignment，还会受到 scene visibility structure 的约束。

---

# 9. Path 3：通过 view-dependent color / SH

理论上：

\[
Pose
\rightarrow
viewing\ direction
\rightarrow
SH
\rightarrow
color
\rightarrow
Loss
\]

所以 spherical harmonics 也能产生 Pose gradient。

但是 Energy-GS **主动不使用这条路径**。

作者观察到：

- SH 对 viewpoint 的变化通常比较平滑
- SH coefficients 本身又是 learnable 的
- 小的 color discrepancy 很容易被 SH 自己拟合
- 当 Pose 已接近 GT 时，这条 gradient 几乎没有有效帮助
- color learning rate 较高时，甚至可能让 Pose 在 GT 附近震荡

因此 Energy-GS 的思想是：

```text
不要让 Pose error
        ↓
被 SH appearance 参数快速吸收
```

而尽量让：

```text
geometry / projection / visibility mismatch
        ↓
产生有效 Pose gradient
```

---

# 10. 为什么要冻结 Gaussian position？

这是 Energy-GS 最重要的设计之一。

普通训练中：

```text
Pose 在变
Gaussian xyz 也在变
Gaussian scale 也在变
Gaussian 数量也可能变
```

此时 reconstruction error 可以有很多解释方式。

例如真正的问题是：

```text
camera 偏右
```

但 optimizer 可能选择：

```text
Gaussian 往左移动
```

两者都可能让当前 RGB loss 下降。

这形成典型的 **Pose / Geometry ambiguity**。

因此 Energy-GS 在初始 Pose alignment 阶段：

\[
\boxed{
\text{Gaussian xyz fixed}
}
\]

使 optimizer 更难通过 geometry deformation 来逃避 Pose correction。

核心思想是：

> **在 Pose 还没对齐之前，不要给 scene representation 过大的几何自由度。**

---

# 11. 为什么初期关闭 Densification？

标准 3DGS 会执行：

- clone
- split
- prune

这些操作会动态改变 Gaussian primitive 数量。

例如：

```text
iteration k

g1 g2 g3 g4

       ↓ split

iteration k+1

g1 g2 g2' g2'' g3 g4
```

于是：

- 当前 pixel / tile 参与 rasterization 的 primitive 集合改变
- gradient landscape 改变
- Pose gradient 可能发生不连续抖动

Energy-GS 因此在 Pose alignment 初期关闭 densification。

等 Pose 足够稳定后，再重新启用。

---

## 11.1 注意：位置不更新，但位置梯度仍然需要

论文有一个重要工程细节：

> Gaussian position 在 pose-alignment stage 不更新，但 position gradient 仍然保留。

原因是标准 3DGS 的 densification criterion 会利用 Gaussian position gradient 判断：

- 哪些 Gaussian 需要 clone
- 哪些 Gaussian 需要 split

因此：

```text
xyz:
    optimizer update = OFF

xyz.grad:
    backward = ON
```

---

# 12. Image Energy：如何实现 coarse-to-fine？

NeRF / BARF 可以通过 positional encoding frequency 做：

```text
low-frequency
    ↓
high-frequency
```

但 3DGS 是 rasterization-based representation，没有类似的 frequency knob。

Energy-GS 的办法是：

> **直接修改训练目标图像。**

---

# 13. 使用 SVD 分解 GT 图像

对于图像：

\[
I
\]

进行 Singular Value Decomposition：

\[
I = U\Sigma V^T
\]

其中：

\[
\Sigma
=
\text{diag}
(
\sigma_1,
\sigma_2,
\dots,
\sigma_n
)
\]

且：

\[
\sigma_1
\ge
\sigma_2
\ge
\dots
\ge
\sigma_n
\]

定义 image energy：

\[
E =
\|I\|_F^2
=
\sum_{i=1}^{n}\sigma_i^2
\]

只保留前 \(l_v\) 个 singular components，可以得到：

\[
I_E(l_v)
=
U_{l_v}
\Sigma_{l_v}
V_{l_v}^T
\]

也可以写成：

\[
I_E(l_v)
=
\sum_{i=1}^{l_v}
u_i\sigma_i v_i^T
\]

---

# 14. Energy level 的含义

当：

\[
l_v
\]

较小时，只包含图像中最主要的低秩结构。

视觉上通常表现为：

- overall brightness
- 大尺度结构
- dominant texture distribution
- coarse appearance

随着：

\[
l_v \uparrow
\]

逐渐恢复：

- 边缘
- texture
- small details
- high-frequency visual content

因此训练 target 从：

```text
coarse Energy Image
        ↓
more detailed Energy Image
        ↓
near-original image
        ↓
full RGB GT
```

---

# 15. Energy-GS 的监督目标

普通 3DGS：

\[
L =
L_{\text{photo}}
(
\mathcal{R}(G,T),
I_{\text{GT}}
)
\]

Energy-GS：

\[
L =
L_{\text{photo}}
(
\mathcal{R}(G,T),
I_E(\alpha)
)
\]

其中：

\[
I_E(\alpha)
\]

由当前 training progress 控制，并逐渐趋向原始 GT。

因此：

```python
rgb = render(gaussians, pose)

target = energy_image(
    gt,
    progress
)

loss = photometric_loss(
    rgb,
    target
)
```

关键变化并不在 rasterizer，而在 **supervision target**。

---

# 16. 为什么这种 coarse-to-fine 对 Pose 有帮助？

假设 camera pose 偏差较大。

直接使用 full RGB 时：

```text
复杂纹理
小结构
边缘
反光
重复纹理
局部细节
```

会同时产生大量局部 gradient。

这些 gradient 的方向不一定一致，很容易形成局部最优。

而 coarse image 中：

```text
大轮廓
主要亮度分布
全局结构
```

占主导。

此时 optimizer 更容易先解决：

```text
整体 translation
整体 rotation
大尺度 projection mismatch
```

当 camera 已经接近正确位置，再逐渐加入高频信息进行 fine alignment。

核心逻辑是：

\[
\boxed{
\text{先扩大 Pose optimization 的 capture range}
\rightarrow
\text{再提高 alignment precision}
}
\]

---

# 17. 为什么 SVD 有效，但不要把它简单等价成 Fourier low-pass

需要区分两个概念。

SVD：

\[
I = U\Sigma V^T
\]

是 **low-rank decomposition**。

Fourier / Gaussian blur：

是 **spatial-frequency filtering**。

因此：

\[
\text{SVD rank}
\neq
\text{严格意义上的 frequency}
\]

但在自然图像中，最大的 singular components 往往包含 dominant global structure，因此从视觉效果上可以形成适合 coarse-to-fine optimization 的监督序列。

Energy-GS 真正需要的并不是“严格频率分解”，而是：

> **构造一个连续的、从简单结构到完整图像的 supervision curriculum。**

---

# 18. Densification 的开启由 Energy progress 控制

Energy-GS 不简单使用固定 iteration 打开 densification，而是利用 image-energy progress 作为 Pose 是否进入较稳定阶段的 proxy。

定义：

\[
s
=
\min
\{
step
\mid
l_v(step)>L
\}
\]

当 image energy level 超过阈值 \(L\) 后，再启动 densification。

论文设置包括：

- Synthetic：\(L=20\)
- Mip-NeRF 360：\(L=50\)

核心思想不是这些具体数字，而是：

> **scene complexity 的释放，应和 Pose optimization 的进度关联。**

---

# 19. 完整训练流程

可以把 Energy-GS 抽象成下面的训练过程。

```text
Initial noisy camera poses
           +
Initial Gaussian primitives
           │
           ▼
Parameterize camera pose in se(3)
           │
           ▼
Fix first camera as anchor
           │
           ▼
Freeze Gaussian xyz
Disable early densification
           │
           ▼
Generate low-energy GT image
           │
           ▼
Render 3DGS
           │
           ▼
Photometric loss
           │
           ├──────────────► optimize Pose
           │
           └──────────────► optimize allowed GS attributes
           │
           ▼
Gradually increase image energy
           │
           ▼
Pose becomes increasingly aligned
           │
           ▼
Energy level > threshold
           │
           ▼
Enable densification
           │
           ▼
Fine reconstruction + Pose refinement
```

---

# 20. 用伪代码表示

```python
pose_delta = Parameter(torch.zeros(num_cameras, 6))

# anchor
pose_delta[0].requires_grad_(False)

# early pose-alignment stage
gaussians.xyz.requires_grad_(False)

for step in range(num_steps):

    # ----------------------------
    # Camera
    # ----------------------------
    T = se3_exp(pose_delta) @ T_init

    # ----------------------------
    # Progressive supervision
    # ----------------------------
    energy_level = energy_schedule(step)

    target = build_energy_image(
        gt_image,
        energy_level
    )

    # ----------------------------
    # Differentiable rendering
    # ----------------------------
    rgb = render(
        gaussians,
        T
    )

    # ----------------------------
    # RGB supervision
    # ----------------------------
    loss = photometric_loss(
        rgb,
        target
    )

    loss.backward()

    pose_optimizer.step()
    gaussian_optimizer.step()

    # xyz is not updated during early alignment,
    # but xyz.grad can still be used by densification.

    # ----------------------------
    # Delayed densification
    # ----------------------------
    if energy_level > threshold:
        densify()
```

这是思想级伪代码，不是论文官方代码。

---

# 21. Energy-GS 最本质的 Optimization Philosophy

Energy-GS 可以抽象成：

\[
\boxed{
\text{限制 Scene 的自由度}
+
\text{控制 Supervision 的难度}
}
\]

---

## 第一部分：限制 Scene 的自由度

Pose 未稳定时：

```text
Gaussian geometry 不要随便动
Gaussian 数量不要随便变
appearance 不要过度吸收误差
```

目的：

```text
让真正属于 Pose 的 error
尽量通过 Pose 被修正
```

---

## 第二部分：控制 Supervision 难度

训练早期：

```text
只暴露大结构
```

训练中期：

```text
逐渐加入细节
```

训练后期：

```text
完整 RGB
```

目的：

```text
先得到大的 capture range
再得到高精度 alignment
```

---

# 22. 作者的 1DGS Toy Experiment 想说明什么？

Supplementary Material 用一个简化的 1D Gaussian Splatting signal alignment task 分析优化机制。

核心结论：

> 单独看 reconstruction quality，并不能判断 signal 的空间位置是否正确。

也就是说：

```text
local rendering 好
        ≠
camera pose 对
```

因为：

```text
Gaussian representation
```

本身有能力重新拟合 signal。

这正是 joint Pose + Scene optimization 中最危险的 ambiguity。

因此：

```text
在 Pose alignment 初期
降低 scene representation 的自由度
```

是一个比单纯增加 optimizer tricks 更根本的解决思路。

---

# 23. 为什么作者认为两项策略缺一不可？

Supplementary Material 的 ablation 表明：

```text
稳定 Gaussian / Pose gradient
            +
Energy coarse-to-fine
```

需要同时存在。

只做：

```text
freeze Gaussian
```

只能让梯度更稳定，但未必拥有足够大的 Pose capture range。

只做：

```text
energy curriculum
```

如果 Gaussian geometry 仍然可以快速变形，也可能继续吸收 Pose error。

因此两者作用不同：

| 模块 | 解决的问题 |
|---|---|
| Freeze geometry / delayed densification | Pose gradient stability |
| Image-energy curriculum | Pose optimization landscape / local minima |
| se(3) parameterization | 合理的 Pose optimization space |
| Fixed first camera | 消除 global gauge ambiguity |

---

# 24. 这篇论文最值得迁移到其他系统的思想

Energy-GS 最有价值的不一定是 SVD 本身，而是以下三个更普适的原则。

---

## Principle A：避免多个变量竞争解释同一个误差

如果：

```text
Pose
Geometry
Appearance
```

都可以同时解释 RGB mismatch，那么 optimizer 很容易选择错误变量。

因此：

\[
\boxed{
\text{在不同训练阶段控制不同变量的自由度}
}
\]

---

## Principle B：Pose optimization 应该 coarse-to-fine

大的 pose error：

```text
需要 smooth / large-basin objective
```

小的 pose error：

```text
需要 high-frequency / detailed objective
```

所以训练目标最好随着 Pose 收敛逐渐变复杂。

---

## Principle C：模型结构动态变化会破坏 Pose gradient

对于：

- Gaussian densification
- point birth/death
- topology changes
- adaptive primitives

都应该考虑：

> **它是否让 Pose optimization 的 objective 在迭代间发生剧烈变化？**

如果答案是“是”，就应该把结构变化推迟到 Pose 已较稳定之后。

---

# 25. 对工程实现最直接的启示

如果已有粗 Pose，例如来自：

- COLMAP
- VGGT
- MASt3R
- SLAM
- visual odometry

但 Pose 仍有 jitter / drift，可以采用类似 schedule：

### Stage A：建立稳定 Scene seed

```text
Pose fixed
GS 建立初始 coarse representation
```

---

### Stage B：Pose alignment

```text
Pose ON
Gaussian xyz OFF
densification OFF

low-complexity / coarse image supervision
```

---

### Stage C：Pose fine alignment

```text
Pose ON
Gaussian geometry 部分释放

supervision gradually approaches RGB
```

---

### Stage D：Full reconstruction

```text
Pose small LR
Gaussian full optimization
densification ON
full RGB supervision
```

---

# 26. 可进一步改进 Energy-GS 的方向

下面属于基于论文思想的工程延伸，而不是论文原方法。

---

## 26.1 SVD 可替换成其他 coarse-to-fine supervision

可以实验：

```text
SVD rank annealing
vs
Gaussian blur annealing
vs
image pyramid
vs
wavelet decomposition
vs
Fourier low-pass → full frequency
```

关键不是必须使用 SVD，而是：

\[
\boxed{
\text{supervision complexity curriculum}
}
\]

---

## 26.2 Energy schedule 可以自适应 Pose convergence

原方法使用预定义 energy schedule。

Supplementary Material 明确指出，效果对：

```text
energy schedule
        ↕
pose learning rate
```

非常敏感。

如果：

```text
Energy 增长太快
Pose LR 太小
```

Pose 可能提前停止改善。

如果：

```text
Energy 增长太慢
Pose LR 太大
```

Pose 可能先对齐，随后震荡甚至重新 drift。

因此可改成：

```text
根据 Pose gradient magnitude
reprojection consistency
photometric residual
trajectory smoothness
```

动态决定何时增加 energy level。

---

## 26.3 加入 Geometry Constraint

Energy-GS 的一个特点是只依赖 RGB。

如果实际 pipeline 已经存在：

- feature tracks
- optical flow
- depth
- point correspondence
- reprojection constraint

则可以进一步做：

\[
L =
L_{\text{energy-photo}}
+
\lambda_{\text{geo}}L_{\text{geometry}}
\]

这样：

```text
Energy-GS
解决 optimization landscape

Geometry loss
解决 pose observability / correspondence
```

二者是互补的。

---

# 27. 方法的限制

Supplementary Material 明确指出两个主要限制。

---

## 27.1 对初始 Gaussian 数量敏感

Pose-alignment stage 中：

```text
Gaussian position fixed
Gaussian number fixed
```

所以 initial primitives 必须足以 coarse represent scene。

太少：

```text
representation capacity 不够
    ↓
rendering collapse
    ↓
pose drift
```

太多：

```text
scene 容易 overfit
    ↓
Gaussian 吸收 image error
    ↓
pose optimization 停滞
```

因此初始 Gaussian 数量是重要 hyperparameter。

---

## 27.2 Energy schedule 与 Pose LR 强耦合

两个 schedule 不能独立随便设置。

核心关系：

\[
\boxed{
pose\ LR(t)
\leftrightarrow
energy\ level(t)
}
\]

需要一起调节。

---

# 28. 一句话总结

Energy-GS 的真正思想不是：

> “设计一个新的 Pose loss。”

而是：

> **通过限制 Gaussian 在训练早期的自由度，让 RGB error 更诚实地反馈给 Pose；同时把 RGB supervision 从粗到细逐步释放，从而获得更稳定、更大 capture range 的 camera pose refinement。**

可以浓缩为：

```text
Noisy Pose
   ↓
SE(3) optimization
   ↓
freeze Gaussian geometry
   ↓
disable early densification
   ↓
coarse image-energy supervision
   ↓
Pose alignment
   ↓
progressively restore details
   ↓
enable densification
   ↓
fine Pose + high-quality 3DGS
```

---

# 29. 最重要的三个 takeaway

### ① Pose 未稳定时，不要让 Gaussian geometry 太自由

否则：

```text
scene 会吸收 Pose error
```

---

### ② Pose 优化最好采用 coarse-to-fine objective

否则：

```text
full RGB 的复杂局部 gradient
容易把 Pose 拉入 local minima
```

---

### ③ Scene representation 的结构变化应该晚于 Pose stabilization

尤其是：

```text
clone
split
densification
pruning
```

这些操作会改变 gradient landscape。

---

# 30. Source

**Main paper**

Yu Gao, Lutong Su, Ruixiang Huang, Tianji Jiang, Jiadong Tang, Yufeng Yue, Yi Yang.  
*Energy-GS: Image Energy-guided Pose Alignment Gaussian Splatting with redesigned pose gradient flow.*  
CVPR 2026.

CVF Open Access:
https://openaccess.thecvf.com/content/CVPR2026/html/Gao_Energy-GS_Image_Energy-guided_Pose_Alignment_Gaussian_Splatting_with_redesigned_pose_CVPR_2026_paper.html

**Supplementary Material**

用户提供：`CVPR_2026_Energy_GS_Supply.pdf`

---

## 备注

本文档分为两类内容：

1. **论文方法本身**：Energy-guided supervision、固定 Gaussian position、delayed densification、Pose gradient path、SE(3) 优化等。
2. **工程延伸建议**：例如 image pyramid / wavelet 替代 SVD、自适应 energy schedule、加入 reprojection geometry loss 等。这些属于基于 Energy-GS 思想的进一步推导，并非论文原方法。
