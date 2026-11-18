"""Paper-specified attention modules and objective functions."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class GALA(nn.Module):
    """Channel gating followed by independent horizontal and vertical gates."""
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True), nn.Conv2d(hidden, channels, 1), nn.Sigmoid())
        self.horizontal_gate = nn.Sequential(nn.Conv2d(channels, channels, 1),
                                             nn.Sigmoid())
        self.vertical_gate = nn.Sequential(nn.Conv2d(channels, channels, 1),
                                           nn.Sigmoid())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        globally_weighted = features * self.channel_gate(features)
        horizontal = globally_weighted.mean(dim=3, keepdim=True)
        vertical = globally_weighted.mean(dim=2, keepdim=True)
        return (globally_weighted * self.horizontal_gate(horizontal) *
                self.vertical_gate(vertical))


class EFPN(nn.Module):
    def __init__(self, channels: int = 256, num_levels: int = 5,
                 reduction: int = 16) -> None:
        super().__init__()
        self.level_attention = nn.ModuleList(
            GALA(channels, reduction) for _ in range(num_levels))

    def forward(self, features):
        if len(features) != len(self.level_attention):
            raise ValueError("EFPN feature count does not match configured levels")
        return tuple(layer(value) for layer, value in
                     zip(self.level_attention, features))


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.channel_mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(hidden, channels, 1, bias=False))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        average = F.adaptive_avg_pool2d(features, 1)
        maximum = F.adaptive_max_pool2d(features, 1)
        channel = torch.sigmoid(self.channel_mlp(average) + self.channel_mlp(maximum))
        weighted = features * channel
        spatial_input = torch.cat((weighted.mean(1, keepdim=True),
                                   weighted.amax(1, keepdim=True)), dim=1)
        return weighted * torch.sigmoid(self.spatial(spatial_input))


class MKA(nn.Module):
    """SwinInsSeg multi-kernel projection with per-branch CBAM."""
    def __init__(self, in_channels: int, out_channels: int = 256,
                 reduction: int = 16) -> None:
        super().__init__()
        self.project = nn.Conv2d(in_channels, out_channels, 1)
        self.branches = nn.ModuleList(
            nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel,
                                    padding=kernel // 2, bias=False),
                          nn.BatchNorm2d(out_channels), nn.GELU(),
                          CBAM(out_channels, reduction))
            for kernel in (1, 3, 5, 7))
        self.fuse = nn.Conv2d(out_channels * 4, out_channels, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.project(features)
        attended = torch.cat([branch(residual) for branch in self.branches], dim=1)
        return residual + self.fuse(attended)


class CrossFusedAttention(nn.Module):
    """Multi-head cross attention with ViT queries and CNN keys/values."""
    def __init__(self, dim: int = 768, num_heads: int = 12) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.dim, self.num_heads, self.head_dim = dim, num_heads, dim // num_heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    @staticmethod
    def _tokens(value: torch.Tensor) -> tuple[torch.Tensor, tuple | None]:
        if value.ndim == 3:
            return value, None
        if value.ndim != 4:
            raise ValueError("CFA expects BNC or BCHW tensors")
        batch, channels, height, width = value.shape
        return value.flatten(2).transpose(1, 2), (batch, channels, height, width)

    def forward(self, vit_features: torch.Tensor,
                cnn_features: torch.Tensor) -> torch.Tensor:
        queries, shape = self._tokens(vit_features)
        context, _ = self._tokens(cnn_features)
        batch = queries.shape[0]
        def heads(value):
            return value.reshape(batch, -1, self.num_heads,
                                 self.head_dim).transpose(1, 2)
        q, k, v = heads(self.query(queries)), heads(self.key(context)), heads(self.value(context))
        attention = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.head_dim), dim=-1)
        fused = attention @ v
        fused = fused.transpose(1, 2).reshape(batch, -1, self.dim)
        result = self.norm(queries + self.output(fused))
        if shape is None:
            return result
        batch, channels, height, width = shape
        return result.transpose(1, 2).reshape(batch, channels, height, width)


class FeatureAdapter(nn.Module):
    def __init__(self, dim: int = 768, ratio: int = 4) -> None:
        super().__init__()
        hidden = max(dim // ratio, 1)
        self.adapter = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                     nn.Linear(hidden, dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.adapter(features)


class PositionAdapter(nn.Module):
    def __init__(self, dim: int = 768) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        channels_last = positions.ndim == 4 and positions.shape[-1] == self.conv.in_channels
        value = positions.permute(0, 3, 1, 2) if channels_last else positions
        value = self.conv(self.pool(value))
        value = self.norm(value.permute(0, 2, 3, 1))
        return value if channels_last else value.permute(0, 3, 1, 2)


class Mish(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features * torch.tanh(F.softplus(features))


class ECA(nn.Module):
    def __init__(self, channels: int, gamma: int = 2, b: int = 1) -> None:
        super().__init__()
        estimate = int(abs(math.log2(channels) / gamma + b / gamma))
        kernel = estimate if estimate % 2 else estimate + 1
        self.conv = nn.Conv1d(1, 1, max(kernel, 1),
                              padding=max(kernel, 1) // 2, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        descriptor = features.mean(dim=(2, 3), keepdim=False).unsqueeze(1)
        weights = torch.sigmoid(self.conv(descriptor)).squeeze(1).unsqueeze(-1).unsqueeze(-1)
        return features * weights


def samsar_hybrid_loss(logits: torch.Tensor, targets: torch.Tensor,
                       gamma: float = 1e-5) -> torch.Tensor:
    targets = targets.to(dtype=logits.dtype)
    binary_cross_entropy = F.binary_cross_entropy_with_logits(logits, targets)
    probabilities = torch.sigmoid(logits)
    dimensions = tuple(range(1, logits.ndim))
    intersection = (probabilities * targets).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + targets.sum(dim=dimensions)
    dice = 1 - ((2 * intersection + gamma) / (denominator + gamma)).mean()
    return .2 * binary_cross_entropy + .8 * dice


def solov2_training_loss(loss_cate: torch.Tensor, loss_mask: torch.Tensor,
                         mask_weight: float = 3.) -> torch.Tensor:
    return loss_cate + mask_weight * loss_mask

