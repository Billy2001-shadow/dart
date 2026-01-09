import argparse
import logging
import os
import pprint
import random

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataset.nyud import NYUD  # 训练集
from dataset.kitti import get_kitti_loader

from network.dpt import TinyVimDepth
from util.loss import SiLogLoss
from util.metric import eval_depth
from util.config_loader import load_config
from util.utils import init_log, count_parameters

parser = argparse.ArgumentParser(description='TinyVim Depth for Metric Depth Estimation')
parser.add_argument('--config', default='configs/nyud/train.yaml', help='配置文件路径')
args = parser.parse_args()
args = load_config(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def main():
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_directory, exist_ok=True)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    all_args = {**vars(args)}
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    writer = SummaryWriter(args.log_directory)
    cudnn.enabled = True
    cudnn.benchmark = True

    #################################################################### DataLoader ####################################################################
    
    if args.dataset == 'NYUD':
        size = (args.img_w, args.img_h)
        trainset = NYUD(args.trainset_path, 'train', size=size,augment_camera_intrinsics=args.augment_camera_intrinsics)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, pin_memory=True, num_workers=args.workers,
                                 drop_last=True)
        valset = NYUD(args.valset_path, 'val', size=size)
        valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=True)
    elif args.dataset == 'KITTI':
        size = (args.img_size, args.img_size)
        trainloader = get_kitti_loader(args.trainset_path, 'train', size=size) # torch.Size([16, 3, 448, 1472])
        valloader = get_kitti_loader(args.valset_path,'val',size=size)
        
    else:
        raise NotImplementedError
    #################################################################### DataLoader ####################################################################

    ###################################################################  Model Load ####################################################################
    model = TinyVimDepth(max_depth=args.max_depth)  # 将模型移动到 GPU
                         
                         
    if args.pretrained_from:
        checkpoint = torch.load(args.pretrained_from, map_location='cpu')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        #missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        missing_keys, unexpected_keys = model.load_state_dict({k: v for k, v in state_dict.items() if 'pretrained' in k}, strict=False)
        print(f"Loaded pretrained weights from {args.pretrained_from}")
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        # 检查所有参数是否被加载
        total_params = sum(p.numel() for p in model.parameters())
        loaded_params = sum(p.numel() for k, p in model.named_parameters() if k in state_dict)
        print(f"\n参数总数: {total_params}, 已加载参数: {loaded_params} (占比: {loaded_params / total_params:.1%})")

        # 在加载权重后添加以下代码
        print("\n=== Pretrained Weight Verification ===")

    # 根据配置决定是否冻结 backbone
    if getattr(args, "freeze_backbone", False):
        frozen = 0
        for name, param in model.named_parameters():
            if 'pretrained' in name:
                param.requires_grad = False
                frozen += param.numel()
        logger.info(f"Backbone frozen, params frozen: {frozen:,}")

    model.to(device)
    pretrained_params, head_params, total_params = count_parameters(model)
    logger.info(f"Model Parameters Summary:")
    logger.info(f"Pretrained (TinyViM) parameters: {pretrained_params:,}")
    logger.info(f"Depth Head parameters: {head_params:,}")
    logger.info(f"Total parameters: {total_params:,}")
    ####################################################################################################################################################

    ###################################################################  Loss &&  Optimizer  ###########################################################
    criterion = SiLogLoss().to(device)

    backbone_params = [param for name, param in model.named_parameters() if 'pretrained' in name and param.requires_grad]
    head_params = [param for name, param in model.named_parameters() if 'pretrained' not in name and param.requires_grad]

    param_groups = []
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': args.lr, 'name': 'backbone'})
    if head_params:
        param_groups.append({'params': head_params, 'lr': args.lr * 10.0, 'name': 'head'})

    optimizer = AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)

    # 打印可训练参数信息
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'可训练参数: {trainable_params:,} / {total_params:,} ({trainable_params / total_params * 100:.1f}%)')
    #####################################################################################################################################################
    total_iters = args.epochs * len(trainloader)

    previous_best = {'d1': 0, 'd2': 0, 'd3': 0, 'abs_rel': 100, 'sq_rel': 100, 'rmse': 100, 'rmse_log': 100,
                     'log10': 100, 'silog': 100}
    
    # 添加最佳d1值跟踪
    best_d1 = 0.0
    best_d1_epoch = 0

    for epoch in range(args.epochs):
        logger.info(
            '===========> Epoch: {:}/{:}, d1: {:.3f}, d2: {:.3f}, d3: {:.3f}'.format(epoch, args.epochs,
                                                                                     previous_best['d1'],
                                                                                     previous_best['d2'],
                                                                                     previous_best['d3']))
        logger.info('===========> Epoch: {:}/{:}, abs_rel: {:.3f}, sq_rel: {:.3f}, rmse: {:.3f}, rmse_log: {:.3f}, '
                    'log10: {:.3f}, silog: {:.3f}'.format(
            epoch, args.epochs, previous_best['abs_rel'], previous_best['sq_rel'], previous_best['rmse'],
            previous_best['rmse_log'], previous_best['log10'], previous_best['silog']))

        model.train()
        total_loss = 0

        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()

            img, depth, valid_mask = sample['image'].cuda(), sample['depth'].cuda(), sample['valid_mask'].cuda()

            if random.random() < 0.5:
                img = img.flip(-1)
                depth = depth.flip(-1)
                valid_mask = valid_mask.flip(-1)

            pred = model(img)

            loss = criterion(pred, depth, (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            iters = epoch * len(trainloader) + i

            lr = args.lr * (1 - iters / total_iters) ** 0.9

            for g in optimizer.param_groups:
                if g.get('name') == 'backbone':
                    g["lr"] = lr
                else:
                    g["lr"] = lr * 10.0

            writer.add_scalar('train/loss', loss.item(), iters)

            if i % 100 == 0:
                logger.info('Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}'.format(i, len(trainloader),
                                                                           optimizer.param_groups[0]['lr'], loss.item()))

        model.eval()

        results = {'d1': 0.0, 'd2': 0.0, 'd3': 0.0,
                   'abs_rel': 0.0, 'sq_rel': 0.0, 'rmse': 0.0,
                   'rmse_log': 0.0, 'log10': 0.0, 'silog': 0.0}
        nsamples = 0

        for i, sample in enumerate(valloader):
            img, depth, valid_mask = sample['image'].cuda().float(), sample['depth'].cuda()[0], sample['valid_mask'].cuda()[0]

            with torch.no_grad():
                pred = model(img)
                pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]
            
                eigen_crop_mask = torch.zeros_like(depth, dtype=torch.bool, device=depth.device)
                if args.dataset == 'NYUD':
                    eigen_crop_mask[45: 471, 41: 601] = True
                elif args.dataset == 'KITTI':
                    eigen_crop_mask[153:371, 44:1197] = True # # (218, 1153)
                else:
                    raise NotImplementedError
                    
                depth_mask  = (depth >= args.min_depth) & (depth <= args.max_depth)
                valid_mask = eigen_crop_mask & depth_mask

                if valid_mask.sum() < 10:
                    continue

                cur_results = eval_depth(pred[valid_mask], depth[valid_mask])

                for k in results.keys():
                    results[k] += cur_results[k]
                nsamples += 1

        # 计算平均指标
        for k in results.keys():
            results[k] /= nsamples

        logger.info('==========================================================================================')
        logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
        logger.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(
            *tuple([results[k] for k in results.keys()])))
        logger.info('==========================================================================================')
        print()

        for name, metric in results.items():
            writer.add_scalar(f'eval/{name}', metric, epoch)

        # 更新最佳指标
        for k in results.keys():
            if k in ['d1', 'd2', 'd3']:
                previous_best[k] = max(previous_best[k], results[k])
            else:
                previous_best[k] = min(previous_best[k], results[k])
                
        # 检查是否为最佳d1
        current_d1 = results['d1']
        if current_d1 > best_d1:
            best_d1 = current_d1
            best_d1_epoch = epoch
            logger.info(f'✨ 发现新的最佳d1: {current_d1:.4f} (epoch: {epoch})')
            
            # 保存最佳d1模型
            best_checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'd1': best_d1,
                'previous_best': previous_best,
                'args': args
            }     
            # 同时保存为best_d1.pth（覆盖之前的最佳模型）
            torch.save(best_checkpoint, os.path.join(args.save_path, 'best_d1.pth'))
            logger.info(f'💾 已更新best_d1.pth (当前最佳: {current_d1:.4f})')

        # 保存当前epoch的模型
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'current_metrics': results,
            'previous_best': previous_best,
            'args': args
        }
        torch.save(checkpoint, os.path.join(args.save_path, f'latest_epoch{epoch}.pth'))
        logger.info(f'📁 已保存epoch {epoch}的模型')

    # 训练结束时输出最佳模型信息
    logger.info(f'🏆 训练完成！最佳d1: {best_d1:.4f} (来自epoch {best_d1_epoch})')
    logger.info(f'最佳模型已保存为: {os.path.join(args.save_path, "best_d1.pth")}')


if __name__ == '__main__':
    main()
    
