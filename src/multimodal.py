"""
Multimodal Vision-Language Integration & Distillation Module
============================================================
Provides:
1. ClinicalTextEncoder: Deep semantic encoder for multilingual free-text radiology reports.
2. MultimodalContrastiveModel: Vision-Language contrastive alignment (InfoNCE) pairing 3D MRI with reports.
3. MultimodalFusionClassifier: Dual-modality cross-attention classifier combining MRI vision & report embeddings.
"""

import sys
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import MultiViewKneeModel, TARGET_COLS


def clean_text_for_embedding(text: str) -> str:
    """
    Normalizes report text by stripping diacritics, lowercasing, and normalizing whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFD', text)
    text = "".join([c for c in text if unicodedata.category(c) != 'Mn'])
    text = text.lower()
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'[^a-z0-9\s\.\,\:\;\-]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


class ClinicalTextEncoder(nn.Module):
    """
    Semantic text encoder tailored for multilingual radiology reports (Spanish, French, Dutch, German, English).
    Tokenizes reports into subword-hashed sequences, applies embedding lookup, BiGRU contextual encoding,
    and clinical self-attention pooling to produce a fixed-dimension dense representation.
    """

    def __init__(
        self,
        vocab_size: int = 20000,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        out_dim: int = 256,
        max_seq_len: int = 128,
        dropout: float = 0.2
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.out_dim = out_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bigru = nn.GRU(
            embed_dim, 
            hidden_dim, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        self.attn_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def _tokenize(self, text_list: List[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hashes text tokens into integer ID sequences with length masking.
        """
        token_ids_batch = []
        masks_batch = []

        for text in text_list:
            cleaned = clean_text_for_embedding(text)
            words = cleaned.split()[:self.max_seq_len]
            if not words:
                words = ['normal']

            # Deterministic hash to vocab space (1 to vocab_size-1, 0 is padding)
            ids = [(abs(hash(w)) % (self.vocab_size - 1)) + 1 for w in words]
            mask = [1.0] * len(ids)

            # Pad up to max_seq_len
            if len(ids) < self.max_seq_len:
                pad_len = self.max_seq_len - len(ids)
                ids = ids + [0] * pad_len
                mask = mask + [0.0] * pad_len

            token_ids_batch.append(ids)
            masks_batch.append(mask)

        ids_tensor = torch.tensor(token_ids_batch, dtype=torch.long, device=device)
        mask_tensor = torch.tensor(masks_batch, dtype=torch.float32, device=device)
        return ids_tensor, mask_tensor

    def forward(self, reports: List[str], device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Encodes a list of report text strings into a dense tensor of shape (B, out_dim).
        """
        if device is None:
            device = next(self.parameters()).device

        input_ids, mask = self._tokenize(reports, device=device)
        embeds = self.embedding(input_ids)  # (B, Seq_Len, embed_dim)

        gru_out, _ = self.bigru(embeds)     # (B, Seq_Len, hidden_dim * 2)

        # Attention pooling with padding mask
        attn_scores = self.attn_fc(gru_out) # (B, Seq_Len, 1)
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=1) # (B, Seq_Len, 1)

        pooled = torch.sum(gru_out * attn_weights, dim=1) # (B, hidden_dim * 2)
        return self.out_proj(pooled) # (B, out_dim)


class MultimodalContrastiveModel(nn.Module):
    """
    Vision-Language Contrastive Alignment Model (MedCLIP-style).
    Aligns 3D multi-view knee MRI volume embeddings with multilingual radiology report embeddings
    using symmetric InfoNCE contrastive loss and learnable temperature scaling.
    """

    def __init__(
        self,
        vision_model: Optional[MultiViewKneeModel] = None,
        text_encoder: Optional[ClinicalTextEncoder] = None,
        latent_dim: int = 256,
        initial_temperature: float = 0.07
    ):
        super().__init__()
        self.vision_model = vision_model if vision_model is not None else MultiViewKneeModel(
            num_classes=12, feature_dim=128, fused_dim=256
        )
        self.text_encoder = text_encoder if text_encoder is not None else ClinicalTextEncoder(
            out_dim=256
        )
        self.latent_dim = latent_dim

        # Visual Latent Projection
        vision_feat_dim = self.vision_model.fused_dim
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_feat_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )

        # Text Latent Projection
        text_feat_dim = self.text_encoder.out_dim
        self.text_proj = nn.Sequential(
            nn.Linear(text_feat_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )

        # Learnable logit scale (inverse temperature parameter: 1 / tau)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / initial_temperature))

    def extract_vision_latent(self, image: torch.Tensor) -> torch.Tensor:
        """
        image: (B, 3, D, H, W)
        Returns: normalized visual latent vector (B, latent_dim)
        """
        sag = image[:, 0:1]
        cor = image[:, 1:2]
        ax  = image[:, 2:3]

        sag_feats = self.vision_model.sag_encoder(sag)
        cor_feats = self.vision_model.cor_encoder(cor)
        ax_feats  = self.vision_model.ax_encoder(ax)

        sag_pooled, _ = self.vision_model.sag_attn(sag_feats)
        cor_pooled, _ = self.vision_model.cor_attn(cor_feats)
        ax_pooled,  _ = self.vision_model.ax_attn(ax_feats)

        fused = self.vision_model.fusion(sag_pooled, cor_pooled, ax_pooled)
        z_v = self.vision_proj(fused)
        return F.normalize(z_v, dim=-1)

    def extract_text_latent(self, reports: List[str]) -> torch.Tensor:
        """
        reports: List of strings
        Returns: normalized text latent vector (B, latent_dim)
        """
        text_feat = self.text_encoder(reports)
        z_t = self.text_proj(text_feat)
        return F.normalize(z_t, dim=-1)

    def forward(
        self, 
        image: torch.Tensor, 
        reports: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            z_v: (B, latent_dim)
            z_t: (B, latent_dim)
            logits_matrix: (B, B) cross-modal similarity matrix scaled by temperature
        """
        z_v = self.extract_vision_latent(image)
        z_t = self.extract_text_latent(reports)

        scale = self.logit_scale.exp().clamp(max=100.0)
        logits_matrix = scale * torch.matmul(z_v, z_t.t()) # (B, B)

        return z_v, z_t, logits_matrix

    def compute_contrastive_loss(self, logits_matrix: torch.Tensor) -> torch.Tensor:
        """
        Symmetric InfoNCE cross-entropy loss over the similarity matrix.
        """
        B = logits_matrix.shape[0]
        labels = torch.arange(B, device=logits_matrix.device)

        loss_v2t = F.cross_entropy(logits_matrix, labels)
        loss_t2v = F.cross_entropy(logits_matrix.t(), labels)

        return (loss_v2t + loss_t2v) / 2.0


class MultimodalFusionClassifier(nn.Module):
    """
    Dual-modality cross-attention classifier combining MRI vision features and free-text report embeddings.
    Fuses both modalities via gated cross-attention to predict the 12 target pathologies during multimodal inference.
    """

    def __init__(
        self,
        contrastive_model: MultimodalContrastiveModel,
        num_classes: int = 12,
        fused_dim: int = 256,
        dropout: float = 0.25
    ):
        super().__init__()
        self.contrastive_model = contrastive_model
        latent_dim = contrastive_model.latent_dim

        # Gated fusion of vision and text latents
        self.gate_fc = nn.Sequential(
            nn.Linear(latent_dim * 2, fused_dim),
            nn.Sigmoid()
        )
        self.joint_fc = nn.Sequential(
            nn.Linear(latent_dim * 2, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.LayerNorm(fused_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, num_classes)
        )

    def forward(self, image: torch.Tensor, reports: List[str]) -> torch.Tensor:
        """
        image: (B, 3, D, H, W)
        reports: List of B text strings
        Returns: logits (B, 12)
        """
        z_v = self.contrastive_model.extract_vision_latent(image)
        z_t = self.contrastive_model.extract_text_latent(reports)

        concat = torch.cat([z_v, z_t], dim=-1) # (B, 2 * latent_dim)
        gate = self.gate_fc(concat)
        feat = self.joint_fc(concat)
        fused = gate * feat

        return self.classifier(fused)

