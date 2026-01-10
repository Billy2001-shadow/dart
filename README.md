# dart

## Camera Embedding的生成过程
1. 用固定 K0 在原图分辨率 (H, W) 调用 generate_rays 得到单位射线 (1, H, W, 3)。（单位方向向量）
2. 对每个目标尺度 s（Hs=H/s, Ws=W/s），用 flat_interpolate 或 F.interpolate 将 rays 插值到 (Hs, Ws, 3)。
L2 归一化 rays。
3. 计算 polar=acos(z)、azimuth=atan2(y, x_clipped)。
4. generate_fourier_features(angles, dim=hidden_dim, max_freq=max(Hs, Ws)//2, use_log=True)，得到 (1, Hs, Ws, hidden_dim)。
5. permute → (1, hidden_dim, Hs, Ws)，缓存后供对应尺度的 DAA/SFH 使用。

SFH 预测的 scale 只是对预训练的视差做全局缩放（纠正相机/域差异），它不负责提供米单位的基准上限；否则训练会很不稳定（极小分母→超大深度）。

###  将每个像素对应的视线方向显示编码处理
1) generate_rays 生成的 rays 是什么
你给定相机内参 K（fx, fy, cx, cy）和图像尺寸 (H, W)。
generate_rays 会在像素平面生成网格坐标 (u,v)，用 K 的逆将每个像素中心 (u+0.5, v+0.5, 1) 投影回相机坐标系，得到一个 3D 向量 (x,y,z)。
每个向量都被 L2 归一化，变成“穿过该像素中心的单位方向”，即一条从相机光心发出的射线的方向。这些方向就是 rays，形状 (B, H*W, 3)。
直观地说，rays 是把每个像素对应的视线方向（单位向量）显式编码出来，代表相机的几何信息（内参、视场和畸变模型）。
> rays 代表了每个像素的视线方向（相机内参决定的几何），与图像内容无关。
### 将3D向量转换极坐标形式(比笛卡尔分量更适合做周期性编码)
angles = torch.stack([polar, azimuth], dim=-1)
- polar极角
- azimuth方位角
> polar/azimuth 把视线方向换成角度坐标，适合做周期编码。
### generate_fourier_features
通过将angles(包含极角和方位角)
Fourier 特征把角度映射到高维、多频率的正弦基，使网络能更容易地利用相机几何信息（例如区分不同视场、不同像面位置的畸变），对下游 DAA/SFH 或 decoder 起到“相机先验提示”的作用。

# 启动脚本
CUDA_VISIBLE_DEVICES=0


需要确认一下DepthFisheye中增强后的特征是如何流动的呢？我感觉现在只是在原有特征上进行增强，下层的特征并没有作为下一层的输入呀感觉(done)
- 优化一下layer
需要更新一下evaluate.py


nohup python train.py > unfreeze_dpt.log 2>&1 &  # TODO
nohup python train.py > unfreeze_dpt_daa_sfh.log 2>&1 & 



# 网络结构
## Encoder（TinyViM + 4 个 stage 级 DAA 插入）

Backbone：TinyViM，4 个 stage，分辨率依次 H/4→H/8→H/16→H/32，对应通道 [48, 64, 168, 224]（默认 TinyViM_S）。
相机嵌入：DenseCameraEmbedder（Fourier on polar/azimuth），对四个尺度生成 cam_embed (默认每尺度 256 维)，输入固定内参 K（可用 [fx,fy,cx,cy] 传入）。
DAA：每个 stage 输出后挂一个 DAAStage 残差，输入该 stage 特征 + 对应尺度 cam_embed，输出同形状特征，继续流向下一个 stage/decoder（实现跨 stage 累积）。
冻结策略：训练时可选 freeze_backbone（在 train.py 里），冻结 TinyViM 的预训练参数，只训练 DAA + decoder + SFH（与 DepthFisheye 的“冻结主干 + adapter 微调”思路一致）。

## Decoder（DPTHead）
4 层特征融合（H/4, H/8, H/16, H/32）经 1×1 投影、上/下采样、refine，再 Sigmoid 输出相对视差，最后缩放/反转得到深度。
该头参数可训练（未冻结）。
SFH（ScaleFormerHead）

输入：最深层特征 (H/32, C4) + 相同分辨率的 cam_embed。
通过 transformer decoder + learnable queries 预测全局尺度 scale ∈ (0,1)（sigmoid），对相对视差进行全局缩放；随后做 inversion（max_depth / disp）得到 metric depth。
训练/优化的意图

冻结 Encoder（保持预训练的泛化特征），仅训练 DAA/SFH + decoder，让微调主要发生在少量参数上，减少灾难性遗忘。
DAA 注入相机几何与低秩残差，SFH 提供全局尺度校正，期望在目标数据集上微调后，对其他数据集的泛化性能损失较小。
这与 DepthFisheye 的思路基本一致：冻结 ViT 主干，插入 Distortion-Aware Adapter（参数高效），并用 ScaleFormer Head 预测全局尺度，适配新域（鱼眼/新相机）且减小遗忘。
思路可行性

可靠性：冻结大 backbone + 小 adapter/head 微调，本身就是常见的 PEFT 手段，用于缓解过拟合/遗忘；你的目标（新域性能提升同时保持泛化）与 DepthFisheye 的动机一致。
注意点：
相机固定时，SFH 的收益可能有限；可考虑对 SFH 添加初始化/正则（如 log-scale 初始化为 1），避免缩放过度影响。
DAA 的 cam_dim 与 embedding 方式已对齐 UniDepth v2（Fourier angles），保留几何先验。
如仍担心遗忘，可降低 head 的学习率或对 decoder 做部分冻结。
文字版结构图（训练时，含冻结策略）

输入 RGB → TinyViM stage1 → DAA1(cam H/4) → downsample → stage2 → DAA2(cam H/8) → downsample → stage3 → DAA3(cam H/16) → downsample → stage4 → DAA4(cam H/32) → 输出特征4。
四尺度特征送入 DPTHead → 相对视差。
最深特征 + cam H/32 → SFH → scale → 视差缩放后 inversion（max_depth / disp）→ 深度输出。
训练时：TinyViM 参数冻结；DAA1-4 + DPTHead + SFH 参与优化。


## DAA
现状：DAAStage 设计回顾（参考 daa.py）

输入：feat 形状 (B,C,H,W)，cam 形状 (1 或 B, CE, H, W)，CE 默认 80/256；cam 先扩展到 B。
处理：
flatten 特征为 tokens (B,N,C)；cam 投影到 C 维 (B,N,C)。
低秩查询 q = A@B，形状 (Q,C)（单头）。
注意力得分：att_f = softmax(q @ k_feat / sqrt(C))，att_c = softmax(q @ k_cam / sqrt(C))，再平均 att = 0.5(att_f+att_c)，得到 (B,Q,N)。
重组：delta = att^T @ q → (B,N,C)，proj_out 线性映射（零未初始化，而是默认 init），reshape 回 (B,C,H,W)；输出 feat + delta 残差。
输出：与输入同形状的特征残差，直接加回，作为下游 stage/decoder 输入。
特征融合方式：使用注意力权重融合，cam 仅参与权重计算，value 取自固定的 q；单头、低秩、未归一化处理。

att_f = softmax(q @ k_feat / sqrt(C))
att_c = softmax(q @ k_cam / sqrt(C))
- 查询 q ∈ ℝ^{Q×C} 与键 k_feat ∈ ℝ^{B×C×N} 做内积，相当于计算 query 与每个 token 的相似度
- 最后用 softmax 做归一化，得到对每个 token 的概率分布
- **找出哪些位置的特征与 query 最相似（贡献最大）**，softmax 确保权重为正且和为 1。


为什么图像特征和相机光线特征分别求注意力然后平均？
- 图像特征注意力（att_f）只基于视觉语义；相机光线注意力（att_c）只基于几何/位置信息。
delta = att^T @ q
- 得到的注意力权重 att（B×N×Q 或 B×Q×N）用来重组 query，att^T @ q 相当于把每个 query 的信息按权重分配回 token


改进1：单头 & 平均权重：无多头、多尺度；att_f/att_c 简单平均，无法自适应融合比例。 
改进2：低秩查询固定：q 不随 token/context 调整，表达力有限，可能不足以补偿域差异。
改进3：投影未零初始化/未共享


太简单了：在cross-attention后面直接加了一个linear，cross-attention主要是把特征根据内参进行变换，在cross-attention后面直接加了一个linear（）
- proj_cam、proj_out设计是不是过于简单了呢？