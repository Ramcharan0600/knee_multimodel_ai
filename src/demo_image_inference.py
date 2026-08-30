"""
Demonstration of Image-Only Inference (3D MRI Volumes -> 12 Pathology Predictions)
==================================================================================
Demonstrates how the Vision AI processes pure 3D MRI DICOM image volumes with NO text reports:
1. Resamples 3D volumes across 3 orthogonal anatomical planes (Sagittal, Coronal, Axial).
2. Runs 2.5D ConvNeXt feature extraction on all 16 slices per plane.
3. Applies Slice Attention Pooling to identify key diagnostic slices containing lesions.
4. Fuses orthogonal tri-plane representations and outputs 12 pathology probabilities.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import MultiViewKneeModel, TARGET_COLS
from src.mri_preprocessor import PLANES, SyntheticDICOMGenerator, VolumeResampler


def demonstrate_image_only_inference():
    print("=" * 75)
    print("[VISION ONLY INFERENCE] 3D MRI VOLUMES -> 12 PATHOLOGY PREDICTIONS")
    print("=" * 75)

    # 1. Simulate a patient study with 3D MRI scans across 3 planes
    print("\n[Stage 1] Loading 3D MRI DICOM Volumes Across 3 Anatomical Planes...")
    
    # Mock raw acquired DICOM series with varying slice counts and resolutions
    raw_sag = np.random.randn(24, 320, 320).astype(np.float32) # 24 sagittal slices
    raw_cor = np.random.randn(20, 280, 280).astype(np.float32) # 20 coronal slices
    raw_ax  = np.random.randn(30, 256, 256).astype(np.float32) # 30 axial slices

    print(f"  • Raw Sagittal Series : Shape={raw_sag.shape} (24 slices, 320x320)")
    print(f"  • Raw Coronal Series  : Shape={raw_cor.shape} (20 slices, 280x280)")
    print(f"  • Raw Axial Series    : Shape={raw_ax.shape} (30 slices, 256x256)")

    # 2. Resample all 3 series to standardized format (Depth=16, H=224, W=224)
    print("\n[Stage 2] Standardizing Geometry via VolumeResampler (Percentile Windowing [0, 1])...")
    resampled_sag = VolumeResampler.preprocess_volume(raw_sag, target_depth=16, target_spatial=(224, 224))
    resampled_cor = VolumeResampler.preprocess_volume(raw_cor, target_depth=16, target_spatial=(224, 224))
    resampled_ax  = VolumeResampler.preprocess_volume(raw_ax,  target_depth=16, target_spatial=(224, 224))

    # Stack into tri-plane batch tensor: [B=1, 3 Channels, Depth=16, H=224, W=224]
    tri_plane_volume = np.stack([resampled_sag, resampled_cor, resampled_ax], axis=0) # (3, 16, 224, 224)
    input_tensor = torch.tensor(tri_plane_volume, dtype=torch.float32).unsqueeze(0)    # (1, 3, 16, 224, 224)

    print(f"  • Standardized Tri-Plane Tensor: {list(input_tensor.shape)}")
    print(f"    [Channel 0: Sagittal | Channel 1: Coronal | Channel 2: Axial]")

    # 3. Vision Model Forward Pass
    print("\n[Stage 3] Multi-View Vision Model (ConvNeXt + Slice Attention)...")
    model = MultiViewKneeModel(num_classes=12, feature_dim=128, fused_dim=256)
    model.eval()

    with torch.no_grad():
        logits, attn_maps = model(input_tensor, return_attention=True)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    # 4. Extract Diagnostic Slice Attention Distributions
    sag_weights = attn_maps['sagittal'].cpu().numpy()[0]
    cor_weights = attn_maps['coronal'].cpu().numpy()[0]
    ax_weights  = attn_maps['axial'].cpu().numpy()[0]

    top_sag = int(np.argmax(sag_weights)) + 1
    top_cor = int(np.argmax(cor_weights)) + 1
    top_ax  = int(np.argmax(ax_weights)) + 1

    print("\n[Stage 4] Diagnostic Slice Attention Results (Which slices did the AI focus on?):")
    print(f"  • Sagittal Plane -> Top Focus: Slice #{top_sag:02d} / 16 (Attention Weight: {sag_weights[top_sag-1]:.3f})")
    print(f"  • Coronal Plane  -> Top Focus: Slice #{top_cor:02d} / 16 (Attention Weight: {cor_weights[top_cor-1]:.3f})")
    print(f"  • Axial Plane    -> Top Focus: Slice #{top_ax:02d} / 16 (Attention Weight: {ax_weights[top_ax-1]:.3f})")

    # 5. Output Prediction Table
    print("\n" + "=" * 75)
    print("[STAGE 5] VISION-ONLY PREDICTIONS (12 TARGET PATHOLOGIES)")
    print("=" * 75)
    print(f"{'Target Pathology':<22} | {'Probability':<12} | {'Diagnostic Call':<12} | Visual Bar")
    print("-" * 75)

    for i, target in enumerate(TARGET_COLS):
        prob = float(probs[i])
        call = "[POSITIVE]" if prob >= 0.5 else "[NEGATIVE]"
        
        # Visual progress bar (20 chars)
        bar_len = int(round(prob * 20))
        bar = "#" * bar_len + "-" * (20 - bar_len)
        
        print(f"{target:<22} | {prob:.4f}       | {call:<12} | [{bar}]")

    print("=" * 75 + "\n")


if __name__ == '__main__':
    demonstrate_image_only_inference()
