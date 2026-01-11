import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import Compose

from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop
from auto_fov_fitting import  auto_fov_fitting

class SUNRGBD(Dataset):
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
        
        depth_raw = np.asarray(Image.open(depth_path), dtype=np.int16)
        depth = np.right_shift(depth_raw, 3) | np.left_shift(depth_raw, (16-3))
        depth = depth.astype(np.float32) / 1000.0 # (730,530)
        # if 'kv2/kinect2data' in img_path:
        #     image, depth = auto_fov_fitting(image, depth, raw_fx=529.5, raw_fy=529.5)
        # elif 'kv2/align_kv2' in img_path:
        #     image, depth = auto_fov_fitting(image, depth, raw_fx=1059.004329, raw_fy=1059.004329)
        # elif 'kv1/NYUdata' in img_path:
        #     image, depth = auto_fov_fitting(image, depth, raw_fx=518.857901, raw_fy=519.469611)
        # elif 'kv1/b3dodata' in img_path:
        #     image, depth = auto_fov_fitting(image, depth, raw_fx=520.532, raw_fy=520.7444)
        # elif 'xtion/sun3ddata' in img_path:
        #     image, depth = auto_fov_fitting(image, depth, raw_fx=570.342205, raw_fy=570.342205)
        # elif 'xtion/xtion_align_data' in img_path:
        #     image, depth = auto_fov_fitting(image, depth, raw_fx=570.342224, raw_fy=570.342224)
        # elif 'realsense/lg' in img_path or 'realsense/sa' in img_path:
        #     image, depth = auto_fov_fitting(image, depth, raw_fx=1387.4893798828125, raw_fy=1387.4893798828125)
        # elif 'realsense/sh' in img_path:
        #     image, depth = auto_fov_fitting(image, depth, raw_fx=1383.16845703125, raw_fy=1383.16845703125)
            
        sample = self.transform({'image': image, 'depth': depth})

        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth']) 
                
        sample['valid_mask'] = sample['depth'] > 0
                
        sample['image_path'] = self.filelist[item].split(' ')[0]
        
        return sample

    def __len__(self):
        return len(self.filelist)
    
def get_sunrgbd_loader(data_dir_root,mode, size=(640, 480)):  # 730、530
    dataset = SUNRGBD(data_dir_root, mode,size)
    return DataLoader(dataset, batch_size=1, shuffle=False,num_workers=4,pin_memory=True)
  