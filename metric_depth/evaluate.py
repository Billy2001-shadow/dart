import argparse
import logging
import pprint

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tqdm import tqdm  # 导入 tqdm


from network.dpt import TinyVimDepth
from network.daa import DAAStage
from network.sfh import ScaleFormerHead
from util.metric import eval_depth
from util.utils import init_log,count_parameters

from torch.utils.data import DataLoader
# from dataset.nyu2 import get_nyud_loader
from dataset.nyud import NYUD

from dataset.ibims import get_ibims_loader
from dataset.sunrgbd import get_sunrgbd_loader
from dataset.eth3d import get_eth3d_loader
from dataset.diode import get_diode_loader

parser = argparse.ArgumentParser(description='TinyVim Depth for Metric Depth Estimation')

parser.add_argument('--dataset', default='IBIMS', help='测试数据集') # NYUD IBIMS SUNRGBD ETH3D DIODE
parser.add_argument('--max_depth', type=int, default=10, help='最大深度')
# dpt_freeze /home/ldc/cw/dev/last_dance/dart/metric_depth/exp/nyud/dpt/freeze_backbone_true/latest_epoch96.pth
# dpt_daa_sfh /home/ldc/cw/dev/last_dance/dart/metric_depth/exp/nyud/dpt_daa_sfh/freeze_backbone_true/CADP_wo_residual/latest_epoch46.pth
parser.add_argument('--weights', type=str,default="/home/ldc/cw/dev/last_dance/dart/metric_depth/exp/nyud/dpt/freeze_backbone_true/best_d1.pth", help='权重文件') 

parser.add_argument('--nyu_val', type=str,default="dataset/splits/nyud/val.txt", help='测试数据集索引文件')
parser.add_argument('--ibims_val', type=str,default="dataset/splits/zeroshot/ibims.txt", help='测试数据集索引文件')
parser.add_argument('--sunrgbd_val', type=str,default="dataset/splits/zeroshot/sunrgbd.txt", help='测试数据集索引文件')
parser.add_argument('--eth3d_indoor_val', type=str,default="dataset/splits/zeroshot/eth3d_indoor.txt", help='测试数据集索引文件')
parser.add_argument('--diode_indoor_val', type=str,default="dataset/splits/zeroshot/diode_indoor.txt", help='测试数据集索引文件')
parser.add_argument('--intrinsic', type=list,default=[550.39, 548.55,319.5,239.5], help='测试数据集索引文件')
parser.add_argument('--module', type=str,default='dpt', help='选择模块 dpt | dpt_sfh | dpt_daa | dpt_daa_sfh')
parser.add_argument('--fusion_method', type=str,default='cross_attention', help='选择模块 cross_attention | additive | concat')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
print(f"Using device: {device}") 
def main():
    
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    all_args = {**vars(args)}
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    cudnn.enabled = True
    cudnn.benchmark = True
    
    if args.dataset == 'NYUD':
        img_w,img_h = 640,480
        min_depth,max_depth = 0.01,10
    elif args.dataset == 'IBIMS':
        img_w,img_h = 640,480
        min_depth,max_depth = 0.01,10
    elif args.dataset == 'SUNRGBD':
        img_w,img_h = 640,480
        min_depth,max_depth = 0.01,10
    elif args.dataset == 'ETH3D':
        img_w,img_h = 640,480 # 448,448 ( 0.458,)
        min_depth,max_depth = 0.01,10
    elif args.dataset == 'DIODE':
        img_w,img_h = 640,480   # (768, 1024) 
        min_depth,max_depth = 0.6,80
    else:
        raise NotImplementedError
#################################################################### DataLoader ####################################################################
    size = (img_w, img_h)
    if args.dataset == 'NYUD':
        valset = NYUD(args.nyu_val, 'val', size=size)
        valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=True)
    elif args.dataset == 'IBIMS':
        valloader = get_ibims_loader(args.ibims_val,'val',size=size)
    elif args.dataset == 'SUNRGBD':
        valloader = get_sunrgbd_loader(args.sunrgbd_val,'val',size=size)
    elif args.dataset == 'ETH3D':
        valloader = get_eth3d_loader(args.eth3d_indoor_val,'val',size=size)
    elif args.dataset == 'DIODE':
        valloader = get_diode_loader(args.diode_indoor_val,'val',size=size)
    else:
        raise NotImplementedError
#################################################################### DataLoader ####################################################################

###################################################################  Model Load ####################################################################
    model = TinyVimDepth(
        max_depth=args.max_depth,
        use_daa=args.module in ['dpt_daa', 'dpt_daa_sfh'],
        use_daa_sfh=args.module in ['dpt_sfh', 'dpt_daa_sfh'],
        intrinsic= args.intrinsic if hasattr(args, 'intrinsic') else None,
        fusion_method=args.fusion_method, 
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
    model.to(device) 
    pretrained_params, head_params, total_params = count_parameters(model)
    logger.info(f"Model Parameters Summary:")
    logger.info(f"Pretrained (TinyViM) parameters: {pretrained_params:,}")
    logger.info(f"Depth Head parameters: {head_params:,}")
    logger.info(f"Total parameters: {total_params:,}")
    
    # 统计 DAA 和 SFH 参数
    daa_params = sum(p.numel() for m in model.modules() if isinstance(m, DAAStage) for p in m.parameters())
    sfh_params = sum(p.numel() for m in model.modules() if isinstance(m, ScaleFormerHead) for p in m.parameters())
    logger.info(f"DAA modules parameters: {daa_params:,}")
    logger.info(f"ScaleFormerHead parameters: {sfh_params:,}")
####################################################################################################################################################


#####################################################################################################################################################
    model.eval()
    
    results = {'d1': 0.0, 'd2': 0.0, 'd3': 0.0, 
                'abs_rel': 0.0, 'sq_rel': 0.0, 'rmse': 0.0, 
                'rmse_log': 0.0, 'log10': 0.0, 'silog': 0.0}
    nsamples = 0
    
    for i, sample in enumerate(tqdm(valloader, desc=f"Testing on {args.dataset}", unit="batch")):
        img, depth, valid_mask = sample['image'].cuda().float(), sample['depth'].cuda()[0], sample['valid_mask'].cuda()[0]
        
        with torch.no_grad():
            pred = model(img)
            pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]
        
        eigen_crop_mask = torch.zeros_like(depth, dtype=torch.bool, device=depth.device)
        if args.dataset == 'NYUD':
            eigen_crop_mask[45: 471, 41: 601] = True
            depth_mask  = (depth >= min_depth) & (depth <= max_depth) & eigen_crop_mask
        elif args.dataset == 'KITTI':
            eigen_crop_mask[153:371, 44:1197] = True # # (218, 1153)
            depth_mask  = (depth >= min_depth) & (depth <= max_depth) & eigen_crop_mask
        elif args.dataset == 'IBIMS' or args.dataset == 'SUNRGBD' or args.dataset =='DDAD' or args.dataset == 'ETH3D' or args.dataset == 'DIODE':
            depth_mask = (depth >= min_depth) & (depth <= max_depth)
        else:
            raise NotImplementedError
        valid_mask = valid_mask & depth_mask

        if valid_mask.sum() < 10:
            continue
        
        cur_results = eval_depth(pred[valid_mask], depth[valid_mask])
        
        for k in results.keys():
            results[k] += cur_results[k]
        nsamples += 1
    
    logger.info('==========================================================================================')
    logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
    logger.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(*tuple([v / nsamples for v in results.values()])))
    logger.info('==========================================================================================')
    print()
    
if __name__ == '__main__':
    main()