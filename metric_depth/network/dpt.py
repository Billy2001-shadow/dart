import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model

from .camera_embed import DenseCameraEmbedder
from .daa import DAAStage
from .sfh import ScaleFormerHead
from .util.blocks import FeatureFusionBlock, _make_scratch
from network import tinyvim

def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )

class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )
    def forward(self, x):
        return self.conv_block(x)
    
class DPTHead(nn.Module):
    def __init__(
        self,
        align_channels=48,                   # 统一通道数(decoder阶段)
        out_channels=[48, 64, 168, 224],     # backbone各层特征图的通道数
        use_bn=False,
    ):
        super(DPTHead, self).__init__()
        
        self.scratch = _make_scratch(
            out_channels,
            align_channels,
            groups=1,
            expand=False,
        )

        self.scratch.refinenet1 = _make_fusion_block(align_channels, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(align_channels, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(align_channels, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(align_channels, use_bn)
        
        head_features_1 = align_channels
        head_features_2 = 16
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        
    def forward(self, features, size_h, size_w):
        layer_1, layer_2, layer_3, layer_4 = features
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)   
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])                
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])        
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])        
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)   
        
        out = self.scratch.output_conv1(path_1)                                                 
        out = F.interpolate(out, (size_h, size_w), mode="bilinear", align_corners=True)         
        out = self.scratch.output_conv2(out)                                                    
        
        return out


class TinyVimDepth(nn.Module):
    def __init__(
        self, 
        decoder_align_channels=48,               # decoder 阶段统一映射通道数
        encoder_out_channels=[48, 64, 168, 224], # encoder 各阶段通道数
        use_bn=False, 
        max_depth=10.0,  
        use_daa: bool = True,
        use_daa_sfh: bool = True,
        cam_dims=(256, 256, 256, 256),
    ):
        super(TinyVimDepth, self).__init__()
      
        """创建TinyViM_S模型""" 
        self.pretrained = create_model(  
                        'TinyViM_S',  
                        num_classes=1000,  
                        pretrained=False,  
                        fork_feat=True    
                    )  
        self.depth_head = DPTHead(
            align_channels=decoder_align_channels,
            out_channels=encoder_out_channels,
            use_bn=use_bn,
        )
        self.use_daa = use_daa
        self.use_daa_sfh = use_daa_sfh
        if self.use_daa or self.use_daa_sfh:
            intrinsic = torch.tensor([[525.0, 0.0, 319.5], [0.0, 525.0, 239.5], [0.0, 0.0, 1.0]]) # NYUDs
            self.cam_embedder = DenseCameraEmbedder(intrinsic, cam_dims=cam_dims)
        if self.use_daa:
            self.daa3 = DAAStage(channels=encoder_out_channels[2], cam_dim=cam_dims[2])
            self.daa4 = DAAStage(channels=encoder_out_channels[3], cam_dim=cam_dims[3])
        if self.use_daa_sfh:
            self.sfh = ScaleFormerHead(in_dim=encoder_out_channels[3], cam_dim=cam_dims[3])
        
        self.max_depth = max_depth

    def forward(self, x):
        # 提取四层特征  [b,48,120,160]、[b, 64, 60, 80]、[b, 168, 30, 40]、[b, 224, 15, 20]
        features = list(self.pretrained(x))

        cam16 = cam32 = None
        if self.use_daa:
            _, _, cam16, cam32 = self.cam_embedder(x.shape[-2], x.shape[-1], device=x.device)
            features[2] = self.daa3(features[2], cam16)
            features[3] = self.daa4(features[3], cam32)
        elif self.use_daa_sfh:
            # still need cam for SFH
            _, _, _, cam32 = self.cam_embedder(x.shape[-2], x.shape[-1], device=x.device)

        depth = self.depth_head(
            features=features,
            size_h=x.shape[-2],
            size_w=x.shape[-1],
        ) 

        if self.use_daa_sfh:
            scale = self.sfh(features[3], cam32)  # (B,1)
            depth = depth * scale.view(-1, 1, 1, 1) * self.max_depth
        else:
            depth = depth * self.max_depth
        
        return depth.squeeze(1)
        
    def load_pretrained(self, pretrained_path):  
        """加载预训练权重"""  
        checkpoint = torch.load(pretrained_path, map_location='cpu')  
        
        # 处理不同的权重格式  
        if 'model' in checkpoint:  
            state_dict = checkpoint['model']  
        else:  
            state_dict = checkpoint  
        
        # 加载权重到backbone  
        missing_keys, unexpected_keys = self.pretrained.load_state_dict(  
            state_dict, strict=False  
        )  
        
        print(f"Loaded pretrained weights from {pretrained_path}")  
        if missing_keys:  
            print(f"Missing keys: {missing_keys}")  
        if unexpected_keys:  
            print(f"Unexpected keys: {unexpected_keys}")
