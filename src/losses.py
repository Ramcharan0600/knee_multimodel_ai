"""
Multi-Label Loss Functions for RSNA Knee Multimodal AI
======================================================
Provides:
1. SoftBCEWithLogitsLoss: Multi-label cross-entropy with continuous pseudo-labels & gold sample weighting.
2. AsymmetricLoss: Asymmetric multi-label focal loss for extreme class imbalance.
3. CombinedKneeLoss: Hybrid multi-target loss function.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftBCEWithLogitsLoss(nn.Module):
    """
    Numerically stable multi-label Binary Cross-Entropy loss supporting:
    1. Continuous soft targets (NLP pseudo-probabilities y in [0.0, 1.0])
    2. Hard binary targets (gold-standard annotations y in {0, 1})
    3. Sample weighting (upweighting gold ground-truth studies)
    4. Per-class positive weighting (handling class imbalance)
    5. Label smoothing
    """

    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        gold_weight: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.gold_weight = gold_weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor, 
        has_gold: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        logits: (B, num_classes)
        targets: (B, num_classes)
        has_gold: Optional (B,) with 1.0 for gold studies, 0.0 for pseudo-labeled
        """
        if self.label_smoothing > 0.0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        pos_weight = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        loss = F.binary_cross_entropy_with_logits(
            logits, 
            targets, 
            pos_weight=pos_weight, 
            reduction='none'
        )

        # Sample weighting for gold cases:
        if has_gold is not None and self.gold_weight != 1.0:
            sample_weights = (1.0 + (self.gold_weight - 1.0) * has_gold.float()).unsqueeze(-1)
            loss = loss * sample_weights

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for Multi-Label Classification (ICCV 2021).
    Applies asymmetric focusing (gamma_neg > gamma_pos) and probability margin shifting 
    to down-weight background easy negatives.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        gold_weight: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.gold_weight = gold_weight
        self.reduction = reduction

    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor, 
        has_gold: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        logits: (B, num_classes)
        targets: (B, num_classes)
        """
        # Probabilities
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        # Asymmetric Clipping Margin for negative probabilities
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Basic CE calculation
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric focusing weights
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1.0 - targets)
            pt = torch.clamp(pt0 + pt1, min=0.0, max=1.0)
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            one_sided_w = torch.pow(torch.clamp(1.0 - pt, min=0.0, max=1.0), one_sided_gamma)
            loss = loss * one_sided_w

        loss = -loss

        if has_gold is not None and self.gold_weight != 1.0:
            sample_weights = (1.0 + (self.gold_weight - 1.0) * has_gold.float()).unsqueeze(-1)
            loss = loss * sample_weights

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# Alias for backward compatibility
AsymmetricFocalLoss = AsymmetricLoss


class CombinedKneeLoss(nn.Module):
    """
    Hybrid loss combining Soft BCE and Asymmetric Focal Loss.
    """

    def __init__(
        self,
        bce_weight: float = 0.7,
        focal_weight: float = 0.3,
        gold_weight: float = 2.0
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.soft_bce = SoftBCEWithLogitsLoss(gold_weight=gold_weight)
        self.asym_loss = AsymmetricLoss(gold_weight=gold_weight)

    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor, 
        has_gold: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        loss_bce = self.soft_bce(logits, targets, has_gold)
        loss_focal = self.asym_loss(logits, targets, has_gold)
        return self.bce_weight * loss_bce + self.focal_weight * loss_focal
