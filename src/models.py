"""
Deep Learning Vision Architectures for RSNA Knee Multimodal AI
==============================================================
Provides:
1. SliceAttentionPooling: Attention-based diagnostic slice weighting.
2. PlaneFeatureExtractor: 2.5D multi-scale residual convolutional backbone with SE attention.
3. CrossPlaneFusion: Multi-view tri-plane gated fusion layer.
4. MultiViewKneeModel: Complete multi-plane end-to-end knee pathology detection model.
5. SingleViewKneeModel: Single-plane specialized vision model.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is accessible
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 12 Target Pathologies
TARGET_COLS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]


class SqueezeExcitation2D(nn.Module):
    """
    Squeeze-and-Excitation channel attention block for 2D spatial feature maps.
    Recalibrates channel-wise feature responses adaptively.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced_channels = max(8, channels // reduction)
        self.fc1 = nn.Conv2d(channels, reduced_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(reduced_channels, channels, kernel_size=1)
        self.act = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.fc1(scale)
        scale = self.act(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        return x * scale


class ConvNeXtBlock2D(nn.Module):
    """
    Modern 2D ConvNeXt-style residual block with depthwise 7x7 conv,
    LayerNorm, 1x1 inverted bottleneck, and GELU activation.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.dwconv = nn.Conv2d(
            in_channels, in_channels, kernel_size=7, stride=stride, 
            padding=3, groups=in_channels, bias=False
        )
        self.norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)
        self.pwconv1 = nn.Conv2d(in_channels, out_channels * 2, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        self.se = SqueezeExcitation2D(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=1, num_channels=out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.dwconv(x)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        out = self.se(out)
        return out + res


class PlaneFeatureExtractor(nn.Module):
    """
    Extracts deep visual features from 2D slices across depth D.
    Input: (B, 1, D, H, W) or (B * D, 1, H, W)
    Output: (B, D, feature_dim)
    """
    def __init__(self, in_channels: int = 1, feature_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.feature_dim = feature_dim

        # Stem: Initial downsampling from (H, W) -> (H/4, W/4)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=4, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=32),
            nn.GELU()
        )

        # Stage 1: (H/4, W/4) -> 64 channels
        self.stage1 = nn.Sequential(
            ConvNeXtBlock2D(32, 64, stride=1),
            ConvNeXtBlock2D(64, 64, stride=1)
        )

        # Stage 2: (H/4, W/4) -> (H/8, W/8) -> 128 channels
        self.stage2 = nn.Sequential(
            ConvNeXtBlock2D(64, 128, stride=2),
            ConvNeXtBlock2D(128, 128, stride=1)
        )

        # Stage 3: (H/8, W/8) -> (H/16, W/16) -> feature_dim channels
        self.stage3 = nn.Sequential(
            ConvNeXtBlock2D(128, feature_dim, stride=2),
            ConvNeXtBlock2D(feature_dim, feature_dim, stride=1)
        )

        # Global spatial pooling and projection
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, D, H, W)
        Returns: (B, D, feature_dim)
        """
        B, C, D, H, W = x.shape
        x_slices = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)

        feat = self.stem(x_slices)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)

        feat = self.global_pool(feat).flatten(1)
        feat = self.norm(feat)
        feat = self.dropout(feat)

        return feat.view(B, D, self.feature_dim)


class SliceAttentionPooling(nn.Module):
    """
    Attention-based diagnostic slice weighting module.
    Learns an attention distribution alpha_i over the D slices:
        alpha_i = softmax(w^T tanh(W * h_i + b))
        pooled_vector = sum(alpha_i * h_i)
    """
    def __init__(self, feature_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, D, feature_dim)
        Returns:
            pooled: (B, feature_dim)
            weights: (B, D) attention distribution over slices
        """
        scores = self.proj(x)
        weights = F.softmax(scores, dim=1)  # (B, D, 1)
        pooled = torch.sum(x * weights, dim=1)
        return pooled, weights.squeeze(-1)


class CrossPlaneFusion(nn.Module):
    """
    Fuses feature representations from the three orthogonal planes (Sagittal, Coronal, Axial).
    """
    def __init__(self, feature_dim: int = 256, fused_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        in_dim = feature_dim * 3
        self.fusion_fc = nn.Sequential(
            nn.Linear(in_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.gate_fc = nn.Sequential(
            nn.Linear(in_dim, fused_dim),
            nn.Sigmoid()
        )
        self.out_proj = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, sag: torch.Tensor, cor: torch.Tensor, ax: torch.Tensor) -> torch.Tensor:
        """
        sag, cor, ax: each (B, feature_dim)
        Returns: (B, fused_dim)
        """
        concat = torch.cat([sag, cor, ax], dim=-1)
        feat = self.fusion_fc(concat)
        gate = self.gate_fc(concat)
        gated = feat * gate
        return self.out_proj(gated)


class MultiViewKneeModel(nn.Module):
    """
    Complete Multi-View Tri-Plane Deep Learning Model for RSNA Knee Pathology Detection.
    """
    def __init__(
        self,
        num_classes: int = 12,
        feature_dim: int = 256,
        fused_dim: int = 512,
        dropout: float = 0.25,
        share_encoders: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.fused_dim = fused_dim
        self.share_encoders = share_encoders

        if share_encoders:
            shared = PlaneFeatureExtractor(in_channels=1, feature_dim=feature_dim, dropout=dropout)
            self.sag_encoder = shared
            self.cor_encoder = shared
            self.ax_encoder = shared
        else:
            self.sag_encoder = PlaneFeatureExtractor(in_channels=1, feature_dim=feature_dim, dropout=dropout)
            self.cor_encoder = PlaneFeatureExtractor(in_channels=1, feature_dim=feature_dim, dropout=dropout)
            self.ax_encoder = PlaneFeatureExtractor(in_channels=1, feature_dim=feature_dim, dropout=dropout)

        self.sag_attn = SliceAttentionPooling(feature_dim=feature_dim)
        self.cor_attn = SliceAttentionPooling(feature_dim=feature_dim)
        self.ax_attn = SliceAttentionPooling(feature_dim=feature_dim)

        self.fusion = CrossPlaneFusion(feature_dim=feature_dim, fused_dim=fused_dim, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.LayerNorm(fused_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, num_classes)
        )

    def forward(
        self, 
        image: Optional[torch.Tensor] = None,
        sagittal: Optional[torch.Tensor] = None,
        coronal: Optional[torch.Tensor] = None,
        axial: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if image is not None and image.ndim == 5:
            sag = image[:, 0:1, :, :, :]
            cor = image[:, 1:2, :, :, :]
            ax  = image[:, 2:3, :, :, :]
        else:
            sag = sagittal
            cor = coronal
            ax = axial

        if sag is None or cor is None or ax is None:
            raise ValueError("Must provide either a 5D `image` tensor or all 3 planes (`sagittal`, `coronal`, `axial`).")

        # 1. Extract slice features: (B, D, feature_dim)
        sag_feats = self.sag_encoder(sag)
        cor_feats = self.cor_encoder(cor)
        ax_feats  = self.ax_encoder(ax)

        # 2. Slice Attention Pooling: (B, feature_dim), (B, D)
        sag_pooled, sag_weights = self.sag_attn(sag_feats)
        cor_pooled, cor_weights = self.cor_attn(cor_feats)
        ax_pooled,  ax_weights  = self.ax_attn(ax_feats)

        # 3. Cross-Plane Fusion: (B, fused_dim)
        fused = self.fusion(sag_pooled, cor_pooled, ax_pooled)

        # 4. Classification: (B, num_classes)
        logits = self.classifier(fused)

        if return_attention:
            attn_dict = {
                'sagittal': sag_weights,
                'coronal': cor_weights,
                'axial': ax_weights
            }
            return logits, attn_dict

        return logits


class SingleViewKneeModel(nn.Module):
    """
    Single-plane specialized model.
    """
    def __init__(
        self,
        num_classes: int = 12,
        feature_dim: int = 256,
        dropout: float = 0.25
    ):
        super().__init__()
        self.encoder = PlaneFeatureExtractor(in_channels=1, feature_dim=feature_dim, dropout=dropout)
        self.attn_pool = SliceAttentionPooling(feature_dim=feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        pooled, _ = self.attn_pool(feats)
        return self.classifier(pooled)
