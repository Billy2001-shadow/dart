import math
import torch
import torch.nn as nn


class DAAStage(nn.Module):
    """
    Distortion-Aware Adapter per stage (parallel residual).
    Uses low-rank query (A,B) to attend to image tokens and camera embeddings.
    """

    def __init__(self, channels: int, cam_dim: int = 81, num_queries: int = 64, rank: int = 8):
        super().__init__()
        self.A = nn.Parameter(torch.randn(num_queries, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, channels) * 0.01)
        self.proj_cam = nn.Linear(cam_dim, channels, bias=False)
        self.proj_out = nn.Linear(channels, channels)
        nn.init.zeros_(self.proj_out.weight)

    def forward(self, feat: torch.Tensor, cam: torch.Tensor):
        """
        feat: (B, C, H, W)
        cam:  (1 or B, CE, H, W)
        """
        B, C, H, W = feat.shape
        N = H * W

        tokens = feat.flatten(2).transpose(1, 2)  # (B, N, C)
        cam = cam.expand(B, -1, -1, -1).flatten(2).transpose(1, 2)  # (B, N, CE)

        q = self.A @ self.B  # (Q, C)
        k_feat = tokens.transpose(1, 2)  # (B, C, N)
        k_cam = self.proj_cam(cam).transpose(1, 2)  # (B, C, N)

        att_f = torch.softmax((q @ k_feat) / math.sqrt(C), dim=-1)  # (B, Q, N)
        att_c = torch.softmax((q @ k_cam) / math.sqrt(C), dim=-1)
        att = 0.5 * (att_f + att_c)

        delta = att.transpose(1, 2) @ q  # (B, N, C)
        delta = self.proj_out(delta)
        delta = delta.transpose(1, 2).reshape(B, C, H, W)

        return feat + delta
