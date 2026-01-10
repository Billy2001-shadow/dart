import cv2
import os
import numpy as np
import matplotlib
cmap = matplotlib.colormaps.get_cmap('Spectral_r')

depth_path = "/data/SM4Depth/Zero_shot_Datasets/iBims1/ibims1_core_raw/depth/storageroom_01.png"

# 1. 读取深度图
depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype('float32') * 50.0 / 65535  # in meters

# 2. 创建有效深度掩码（非零值）
valid_mask = depth > 0
valid_depth = depth[valid_mask]

if len(valid_depth) > 0:
    # 3. 只对有效深度进行归一化
    valid_min = valid_depth.min()
    valid_max = valid_depth.max()
    
    print(f"有效深度范围: [{valid_min:.3f}, {valid_max:.3f}] 米")
    print(f"有效像素比例: {valid_mask.sum()/depth.size:.2%}")
    
    # 4. 归一化有效深度到[0, 255]
    normalized_depth = np.zeros_like(depth)
    normalized_depth[valid_mask] = (valid_depth - valid_min) / (valid_max - valid_min) * 255.0
    normalized_depth = normalized_depth.astype(np.uint8)
    
    # 5. 应用颜色映射
    colored_depth = np.zeros((*depth.shape, 3), dtype=np.uint8)
    colored_depth[valid_mask] = (cmap(normalized_depth[valid_mask]/255.0)[:, :3] * 255)[:, ::-1]
    
    # 可选：将无效区域标记为特定颜色（如黑色）
    colored_depth[~valid_mask] = [0, 0, 0]  # 黑色表示无效
    
else:
    print("警告：没有有效的非零深度值！")
    colored_depth = np.zeros((*depth.shape, 3), dtype=np.uint8)

# 6. 保存结果
os.makedirs("./vis_depth", exist_ok=True)
output_path = os.path.join("./vis_depth", os.path.splitext(os.path.basename(depth_path))[0] + '.png')
cv2.imwrite(output_path, colored_depth)
print(f"已保存: {output_path}")