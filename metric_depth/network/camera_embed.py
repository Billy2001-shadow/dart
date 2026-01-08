import torch
import torch.nn as nn

from .util.geometric import generate_rays
from .util.sht import rsh_cart_8


class DenseCameraEmbedder(nn.Module):
    """
    Generate dense camera embeddings (SHT order 8, CE=81) for four pyramid levels.
    K is assumed fixed; we scale fx, fy, cx, cy by the stage downsample ratio.
    """

    def __init__(self, intrinsic: torch.Tensor):
        super().__init__()
        self.register_buffer("K0", intrinsic.clone())
        self.cache = {}

    def _scale_K(self, scale: int):
        K = self.K0.clone()
        K[0, 0] /= scale
        K[1, 1] /= scale
        K[0, 2] /= scale
        K[1, 2] /= scale
        return K

    def _embed(self, H: int, W: int, scale: int, device):
        key = (H, W, scale)
        if key in self.cache:
            return self.cache[key].to(device)
        Ks = self._scale_K(scale).to(device)
        rays, _ = generate_rays(Ks[None], (H // scale, W // scale))
        rays = rays.view(1, H // scale, W // scale, 3)
        rays = rays / rays.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        emb = rsh_cart_8(rays)  # (1, Hs, Ws, CE)
        emb = emb.permute(0, 3, 1, 2).contiguous()  # (1, CE, Hs, Ws)
        self.cache[key] = emb.detach()
        return emb

    def forward(self, H: int, W: int, device=None):
        device = device or self.K0.device
        return (
            self._embed(H, W, 4, device),
            self._embed(H, W, 8, device),
            self._embed(H, W, 16, device),
            self._embed(H, W, 32, device),
        )
