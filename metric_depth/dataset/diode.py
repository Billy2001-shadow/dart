import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import Compose

from dataset.transform import Resize, NormalizeImage, PrepareForNet
from auto_fov_fitting import  auto_fov_fitting

class DIODE(Dataset):
    def __init__(self, filenames_file,size=(224,224)):
        with open(filenames_file, 'r') as f:
            self.filenames = f.readlines()
        self.transform = Compose([
            Resize(
                width=size[0],
                height=size[1],
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([]))

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        
        image_path = sample_path.split()[0]
        depth_path = sample_path.split()[1]
        depth_mask_path = sample_path.split()[2]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        depth = np.load(depth_path)  # (768, 1024, 1) in meters
        depth = depth.squeeze()  #  (768, 1024)
        eval_mask = np.load(depth_mask_path)
        eval_mask = eval_mask.astype(bool) 
        
        sample = dict(image=image, depth=depth)
        sample = self.transform(sample)
        
        sample['valid_mask'] = eval_mask
        sample['image_path'] = image_path
        return sample

    def __len__(self):
        return len(self.filenames)


def get_diode_loader(data_dir_root,mode,size=(224, 224)):
    dataset = DIODE(data_dir_root, size)
    return DataLoader(dataset, batch_size=1, shuffle=False,num_workers=4,pin_memory=True)
  