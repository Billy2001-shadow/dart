import numpy as np
import random
import cv2

def new_random_crop(img, depth, mask=None, height=None, width=None):
    """
    随机裁剪函数，同时处理图像、深度图和可选的掩码
    """
    if height is None or width is None:
        raise ValueError("height and width must be provided")
    
    if mask is not None:
        assert img.shape[0] == depth.shape[0] == mask.shape[0]
        assert img.shape[1] == depth.shape[1] == mask.shape[1]
    else:
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
    
    if img.shape[0] >= height and img.shape[1] >= width:
        y = random.randint(0, img.shape[0] - height)
        x = random.randint(0, img.shape[1] - width)
        
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width] if depth.ndim == 2 else depth[y:y + height, x:x + width, :]
        
        if mask is not None:
            mask = mask[y:y + height, x:x + width] if mask.ndim == 2 else mask[y:y + height, x:x + width, :]
        
    elif img.shape[1] >= width and img.shape[0] < height:
        x = random.randint(0, img.shape[1] - width)
        
        img = img[:, x:x + width, :]
        depth = depth[:, x:x + width] if depth.ndim == 2 else depth[:, x:x + width, :]
        
        if mask is not None:
            mask = mask[:, x:x + width] if mask.ndim == 2 else mask[:, x:x + width, :]
        
    elif img.shape[1] < width and img.shape[0] >= height:
        y = random.randint(0, img.shape[0] - height)
        
        img = img[y:y + height, :, :]
        depth = depth[y:y + height, :] if depth.ndim == 2 else depth[y:y + height, :, :]
        
        if mask is not None:
            mask = mask[y:y + height, :] if mask.ndim == 2 else mask[y:y + height, :, :]
    
    if mask is not None:
        return img, depth, mask
    else:
        return img, depth

def auto_fov_fitting(image, depth, mask=None, raw_fx=None, raw_fy=None, 
                    target_hfov=0.85756, target_vfov=1.09479, eigen_crop=False):
    '''
    自动调整图像、深度图和可选的掩码的视场
    params:
        NYUD的fov：
        target_hfov: 0.85756 rads ≈ 49.134°
        target_vfov: 1.09479 rads ≈ 62.727°
        mask: 可选的掩码，与image、depth同尺寸
    '''
    if raw_fx is None or raw_fy is None:
        raise ValueError("raw_fx and raw_fy must be provided")
    
    h, w = depth.shape[:2]
    
    # 计算目标尺寸
    fit_height = int(raw_fy * 2 * np.tan(target_hfov / 2))
    fit_width  = int(raw_fx * 2 * np.tan(target_vfov / 2))
    
    # 初始化变量
    fitted_image = image.copy()
    fitted_depth = depth.copy()
    
    if mask is not None:
        fitted_mask = mask.copy()
    
    # 裁剪逻辑
    if fit_height <= h and fit_width <= w:
        if mask is not None:
            fitted_image, fitted_depth, fitted_mask = new_random_crop(
                image, depth, mask, fit_height, fit_width
            )
        else:
            fitted_image, fitted_depth = new_random_crop(
                image, depth, mask=None, height=fit_height, width=fit_width
            )
    else:
        # 如果高度需要填充
        if fit_height > h:
            pad_h_top = fit_height - h
            fitted_image = np.pad(image, ((pad_h_top, 0), (0, 0), (0, 0)), 
                                 'constant', constant_values=(1))
            fitted_depth = np.pad(depth, ((pad_h_top, 0), (0, 0)), 
                                 'constant', constant_values=(0))
            if mask is not None:
                fitted_mask = np.pad(mask, ((pad_h_top, 0), (0, 0)), 
                                    'constant', constant_values=(0))
        else:
            pad_h_top = 0
        
        # 如果宽度需要填充
        if fit_width > w:
            pad_w_left = (fit_width - w) // 2
            pad_w_right = fit_width - w - pad_w_left
            
            fitted_image = np.pad(fitted_image, ((0, 0), (pad_w_left, pad_w_right), (0, 0)), 
                                 'constant', constant_values=(1))
            fitted_depth = np.pad(fitted_depth, ((0, 0), (pad_w_left, pad_w_right)), 
                                 'constant', constant_values=(0))
            if mask is not None:
                fitted_mask = np.pad(fitted_mask, ((0, 0), (pad_w_left, pad_w_right)), 
                                    'constant', constant_values=(0))
    
    # Eigen裁剪
    if eigen_crop:
        # 计算裁剪区域
        crop_top = int(46 * fit_height / 480)
        crop_bottom = int(470 * fit_height / 480)
        crop_left = int(44 * fit_width / 640)
        crop_right = int(608 * fit_width / 640)
        
        # 裁剪
        crop_image = fitted_image[crop_top:crop_bottom, crop_left:crop_right]
        crop_depth = fitted_depth[crop_top:crop_bottom, crop_left:crop_right]
        
        if mask is not None:
            crop_mask = fitted_mask[crop_top:crop_bottom, crop_left:crop_right]
        
        # 调整大小
        ret_image = cv2.resize(crop_image, dsize=(564, 424), interpolation=cv2.INTER_LINEAR)
        ret_depth = cv2.resize(crop_depth, dsize=(564, 424), interpolation=cv2.INTER_NEAREST)
        
        if mask is not None:
            ret_mask = cv2.resize(crop_mask, dsize=(564, 424), interpolation=cv2.INTER_NEAREST)
    else:
        ret_image = fitted_image
        ret_depth = fitted_depth
        if mask is not None:
            ret_mask = fitted_mask
    
    # 根据是否传入mask返回不同的结果
    if mask is not None:
        return ret_image, ret_depth, ret_mask
    else:
        return ret_image, ret_depth

# 使用示例
if __name__ == "__main__":
    # 模拟输入数据
    h, w = 480, 640
    image = np.random.rand(h, w, 3).astype(np.float32)
    depth = np.random.rand(h, w).astype(np.float32)
    mask = np.random.randint(0, 2, (h, w), dtype=np.uint8)  # 二值掩码
    

    # 模拟相机内参
    raw_fx = 550.39 #525.0
    raw_fy = 548.55 #525.0
    
    # 调用函数
    result_image, result_depth, result_mask = auto_fov_fitting(
        image=image,
        depth=depth,
        mask=mask,
        raw_fx=raw_fx,
        raw_fy=raw_fy,
        target_hfov=0.85756,
        target_vfov=1.09479,
        eigen_crop=False
    )
    
    print(f"输入尺寸: image={image.shape}, depth={depth.shape}, mask={mask.shape}")
    print(f"输出尺寸: image={result_image.shape}, depth={result_depth.shape}, mask={result_mask.shape}")
    print(f"掩码唯一值: {np.unique(result_mask)}")

