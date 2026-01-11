import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import Compose

from dataset.transform import Resize, NormalizeImage, PrepareForNet
from auto_fov_fitting import  auto_fov_fitting

class NUSCENES(Dataset):
    def __init__(self, filelist_path, mode, size=(484, 484)):
        
        self.mode = mode
        self.size = size
        
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
    def __getitem__(self, item):
        img_path = self.filelist[item].split(' ')[0]
        depth_path = self.filelist[item].split(' ')[1]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0                                  # (480, 640, 3)
        
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype('float32') / 256.0
        
        # image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
        # depth_gt = np.asarray(Image.open(depth_path), dtype=np.float32) / 256.0
        # image, depth_gt = auto_fov_fitting(image, depth_gt, 1266.4172, 1266.4172)
        # depth_gt = np.expand_dims(depth_gt, axis=2)
        
        
        image, depth = auto_fov_fitting(image, depth, raw_fx=1266.4172, raw_fy=1266.4172,target_hfov=1.428543081969707, target_vfov=0.5118273762007145) # target_fov需要改为kitti的 707.0493, 707.0493
       
            
        sample = self.transform({'image': image, 'depth': depth})

        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth']) 
                
        sample['valid_mask'] = sample['depth'] > 0
                
        sample['image_path'] = self.filelist[item].split(' ')[0]
        
        return sample

    def __len__(self):
        return len(self.filelist)
    
def get_nuscenes_loader(data_dir_root,mode, size=(640, 480)):  # 730、530
    dataset = NUSCENES(data_dir_root, mode,size)
    return DataLoader(dataset, batch_size=1, shuffle=False,num_workers=4,pin_memory=True)
  