import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop
import random

class CameraIntrinsicAugmentation:
    def __init__(self, mode='train', scale_range=(0.9, 1.1), crop_ratio_range=(0.8, 1.0)):
        self.mode = mode
        self.scale_range = scale_range
        self.crop_ratio_range = crop_ratio_range
    
    def __call__(self, sample):
        if self.mode != 'train':
            return sample  # 只在训练时应用扰动
            
        image = sample['image']
        depth = sample['depth']
        
        # 随机缩放因子，模拟焦距变化
        scale_factor = random.uniform(*self.scale_range)
        
        # 随机裁剪比例，模拟主点偏移
        crop_ratio = random.uniform(*self.crop_ratio_range)
        
        # 应用缩放
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        # 使用双线性插值进行缩放
        image_scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        depth_scaled = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # 应用随机裁剪，模拟主点偏移
        crop_h = int(new_h * crop_ratio)
        crop_w = int(new_w * crop_ratio)
        
        # 随机选择裁剪起始点
        start_h = random.randint(0, new_h - crop_h)
        start_w = random.randint(0, new_w - crop_w)
        
        # 执行裁剪
        image_cropped = image_scaled[start_h:start_h+crop_h, start_w:start_w+crop_w]
        depth_cropped = depth_scaled[start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        # 如果需要，可以添加padding来保持原始尺寸
        if crop_h < h or crop_w < w:
            # 计算需要padding的量
            pad_h = h - crop_h
            pad_w = w - crop_w
            
            # 随机分配padding到四周
            top = random.randint(0, pad_h)
            bottom = pad_h - top
            left = random.randint(0, pad_w)
            right = pad_w - left
            
            # 应用padding（使用边缘填充或零填充）
            image_padded = cv2.copyMakeBorder(image_cropped, top, bottom, left, right, 
                                            cv2.BORDER_REFLECT)
            depth_padded = cv2.copyMakeBorder(depth_cropped, top, bottom, left, right,
                                            cv2.BORDER_CONSTANT, value=0)
        else:
            image_padded = image_cropped
            depth_padded = depth_cropped
        
        # 确保最终尺寸与原始一致
        image_final = cv2.resize(image_padded, (w, h), interpolation=cv2.INTER_LINEAR)
        depth_final = cv2.resize(depth_padded, (w, h), interpolation=cv2.INTER_NEAREST)
        
        return {'image': image_final, 'depth': depth_final}

class NYUD(Dataset):
    def __init__(self, filelist_path, mode, size=(640, 480),augment_camera_intrinsics=False):
        
        self.mode = mode
        self.size = size
        self.augment_camera_intrinsics = augment_camera_intrinsics
        
        with open(filelist_path, 'r') as f:
            self.filelist = f.read().splitlines()
        
        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        # 添加相机内参扰动
        if self.augment_camera_intrinsics:
            self.camera_aug = CameraIntrinsicAugmentation(mode=mode)
    def __getitem__(self, item):
        img_path = self.filelist[item].split(' ')[0]
        depth_path = self.filelist[item].split(' ')[1]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype('float32') / 1000.0 # in meters
        # 执行Eigen Crop（高45:471，宽41:601）
        # image = image[45:471, 41:601, :]  # 新尺寸: (426, 560, 3)
        # depth = depth[45:471, 41:601]     # 新尺寸: (426, 560)
        # 在transform之前应用相机内参扰动
        if self.augment_camera_intrinsics:
            sample = self.camera_aug({'image': image, 'depth': depth})
        sample = self.transform({'image': image, 'depth': depth})

        sample['image'] = torch.from_numpy(sample['image']) # torch.Size([3, 480, 640])
        sample['depth'] = torch.from_numpy(sample['depth']) # torch.Size([480, 640])
                
        crop_mask = np.zeros_like(sample['depth'], dtype=bool)
        crop_mask[45:471, 41:601] = True  # 只有这个区域是有效的（Eigen Crop）
        sample['valid_mask'] = (sample['depth'] > 0) & torch.from_numpy(crop_mask)
                
        sample['image_path'] = self.filelist[item].split(' ')[0]
        
        return sample

    def __len__(self):
        return len(self.filelist)
    