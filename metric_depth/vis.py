import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from network.dpt import TinyVimDepth


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--img-path', type=str,default='/home/chenwu/DART/dart/metric_depth/vis.txt', help='输入图像路径或包含图像路径的文本文件')
    parser.add_argument('--input-size', type=int, default=480)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--weights', type=str,default="/home/chenwu/DART/dart/metric_depth/exp/nyud/dpt_daa_sfh/freeze_backbone_false/latest_epoch14.pth", help='权重文件')
    parser.add_argument('--max_depth', type=float, default=10.0)
    parser.add_argument('--module', type=str,default='dpt_daa_sfh', help='选择模块 dpt | dpt_sfh | dpt_daa | dpt_daa_sfh')

    parser.add_argument('--pred-only', type=bool, default=True, help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model = TinyVimDepth(
        max_depth=args.max_depth,
        use_daa=args.module in ['dpt_daa', 'dpt_daa_sfh'],
        use_daa_sfh=args.module in ['dpt_sfh', 'dpt_daa_sfh'],
        intrinsic= args.intrinsic if hasattr(args, 'intrinsic') else None
    )  # 将模型移动到 GPU
    if args.weights:
        checkpoint = torch.load(args.weights, map_location='cpu')  
        if 'model' in checkpoint:  
            state_dict = checkpoint['model']  
        else:  
            state_dict = checkpoint  
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {args.weights}")  
        if missing_keys:  
            print(f"Missing keys: {missing_keys}")  
        if unexpected_keys:  
            print(f"Unexpected keys: {unexpected_keys}")
        # 检查所有参数是否被加载
        total_params = sum(p.numel() for p in model.parameters())
        loaded_params = sum(p.numel() for k, p in model.named_parameters() if k in state_dict)
        print(f"\n参数总数: {total_params}, 已加载参数: {loaded_params} (占比: {loaded_params/total_params:.1%})")
        
        # 在加载权重后添加以下代码
        print("\n=== Pretrained Weight Verification ===")
    model.to(DEVICE) 
    
    
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_image = cv2.imread(filename)
        
        depth = model.infer_image(raw_image, args.input_size)
        
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        if args.pred_only:
            cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), depth)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])
            
            cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), combined_result)