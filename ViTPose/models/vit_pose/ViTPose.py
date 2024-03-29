import torch.nn as nn

from .backbone.vit import ViT
from .head.topdown_heatmap_simple_head import TopdownHeatmapSimpleHead


__all__ = ['ViTPose']


class ViTPose(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super(ViTPose, self).__init__()
        
        backbone_cfg = {k: v for k, v in cfg['backbone'].items() if k != 'type'}
        head_cfg = {k: v for k, v in cfg['keypoint_head'].items() if k != 'type'}
        
        self.backbone = ViT(**backbone_cfg)
        self.keypoint_head = TopdownHeatmapSimpleHead(**head_cfg)

        self.downscale = nn.Conv2d(cfg['backbone']['embed_dim'], cfg['backbone']['embed_dim'], 16)
        self.upscale = [
            nn.ConvTranspose2d(cfg['backbone']['embed_dim'], cfg['backbone']['embed_dim'], kernel_size=16),
            nn.ReLU()]
        self.upscale = nn.Sequential(*self.upscale)

    
    def forward_features(self, x):
        x = x.permute(0, 3, 1, 2).cuda()
        return self.backbone(x)
    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2).cuda()

        vit = self.backbone(x)

        one_d = self.downscale(vit)
        vit_ = self.upscale(one_d)

        hms = self.keypoint_head(vit + vit_)

        # Unsqueeze is consistency with Hourglass
        return hms.unsqueeze(1), one_d.detach()


    def get_one_dimensional(self, x):
        # Shape: torch.Size([1, 64, 1, 1])
        x = x.permute(0, 3, 1, 2).cuda()
        return self.downscale(self.backbone(x))