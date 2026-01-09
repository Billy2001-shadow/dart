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


需要确认一下DepthFisheye中增强后的特征是如何流动的呢？我感觉现在只是在原有特征上进行增强，下层的特征并没有作为下一层的输入呀感觉
需要更新一下evaluate.py