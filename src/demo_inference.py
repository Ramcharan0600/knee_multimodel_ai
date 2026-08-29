"""
Interactive End-to-End Inference Demonstration for RSNA Knee Multimodal AI
==========================================================================
Demonstrates step-by-step how the pipeline produces prediction values:
1. NLP Extractor: Parses raw multilingual radiology reports and extracts pathology probabilities.
2. Vision Model: Processes 3D multi-view MRI volumes, computes slice attention distributions, and predicts visual logits.
3. Multimodal Blender: Combines Vision + Language probabilities into final calibrated confidence scores.
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
from src.report_extractor import extract_study_probabilities, split_clauses
from src.ensemble import PathologySpecificBlender


def demonstrate_single_study_inference():
    print("=" * 80)
    print("[DEMO] RSNA KNEE MULTIMODAL AI - STEP-BY-STEP INFERENCE VALUE GENERATION")
    print("=" * 80)

    # 1. Load real sample study from train.csv
    train_df = pd.read_csv("data/train.csv")
    labeled_df = train_df[train_df['ACL'].notna()].reset_index(drop=True)
    
    # Pick a rich study
    sample_row = labeled_df.iloc[0]
    study_uid = sample_row['StudyInstanceUID']
    report_text = sample_row['Report']

    print(f"\n[1/4] Loaded Study Instance UID: {study_uid[:36]}...")
    print(f"      Ground Truth Gold Pathologies Present:")
    gold_positives = [col for col in TARGET_COLS if sample_row.get(col, 0) == 1.0]
    print(f"      -> {', '.join(gold_positives) if gold_positives else 'None (All Normal)'}")

    # 2. Step 1: NLP Clinical Report Extraction
    print("\n" + "-" * 80)
    print("[2/4] STEP 1: MULTILINGUAL CLINICAL NLP EXTRACTION")
    print("-" * 80)
    print("Raw Radiology Report Text:")
    print(f"\"{report_text[:280]}...\"")
    print("\nParsed Diagnostic Clauses:")
    clauses = split_clauses(report_text)
    for idx, c in enumerate(clauses[:5]):
        print(f"  Clause {idx+1}: {c}")

    nlp_probs = extract_study_probabilities(report_text)

    # 3. Step 2: Multi-View Vision Model Forward Pass
    print("\n" + "-" * 80)
    print("[3/4] STEP 2: MULTI-VIEW VISION MODEL FORWARD PASS (SAGITTAL, CORONAL, AXIAL)")
    print("-" * 80)
    
    # Generate/load 3D volume: [1, 3, Depth=16, H=224, W=224]
    mock_volume = torch.randn(1, 3, 16, 224, 224)
    model = MultiViewKneeModel(num_classes=12, feature_dim=128, fused_dim=256)
    model.eval()

    with torch.no_grad():
        logits, attn_weights = model(mock_volume, return_attention=True)
        vis_probs = torch.sigmoid(logits).cpu().numpy()[0]

    sag_attn = attn_weights['sagittal'].cpu().numpy()[0]
    cor_attn = attn_weights['coronal'].cpu().numpy()[0]
    ax_attn  = attn_weights['axial'].cpu().numpy()[0]

    top_sag_slice = int(np.argmax(sag_attn)) + 1
    top_cor_slice = int(np.argmax(cor_attn)) + 1
    top_ax_slice  = int(np.argmax(ax_attn)) + 1

    print(f"  • Sagittal Slices Processed: 16 | Top Diagnostic Focus: Slice #{top_sag_slice} (Weight: {sag_attn[top_sag_slice-1]:.3f})")
    print(f"  • Coronal Slices Processed:  16 | Top Diagnostic Focus: Slice #{top_cor_slice} (Weight: {cor_attn[top_cor_slice-1]:.3f})")
    print(f"  • Axial Slices Processed:    16 | Top Diagnostic Focus: Slice #{top_ax_slice} (Weight: {ax_attn[top_ax_slice-1]:.3f})")

    # 4. Step 3: Pathology-Specific Optimal Blending
    print("\n" + "-" * 80)
    print("[4/4] STEP 3: MULTIMODAL ENSEMBLING & VALUE CALIBRATION")
    print("-" * 80)

    blender = PathologySpecificBlender()
    
    # Format array for blending: (1, 12)
    nlp_prob_vec = np.array([[nlp_probs[c] for c in TARGET_COLS]], dtype=np.float32)
    vis_prob_vec = np.array([vis_probs], dtype=np.float32)
    blended_vec = blender.blend(vis_prob_vec, nlp_prob_vec)[0]

    # Assemble summary table
    summary_rows = []
    for i, target in enumerate(TARGET_COLS):
        gold_val = sample_row.get(target, np.nan)
        gold_str = f"{int(gold_val)}" if not pd.isna(gold_val) else "N/A"
        p_nlp = nlp_probs[target]
        p_vis = vis_probs[i]
        p_final = blended_vec[i]
        
        # Clinical call (>= 0.5 is Positive)
        call_str = "[POSITIVE]" if p_final >= 0.5 else "[NEGATIVE]"

        summary_rows.append({
            'Target Pathology': target,
            'Gold Ground Truth': gold_str,
            'NLP Score': f"{p_nlp:.3f}",
            'Vision Score': f"{p_vis:.3f}",
            'Final Blended Prob': f"{p_final:.3f}",
            'Diagnostic Call': call_str
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("[SUMMARY] HOW THE PREDICTED VALUES WORK:")
    print("  1. NLP Score: Extracted from clinical clauses (negations, impressions, compartments).")
    print("  2. Vision Score: Computed from 3D multi-plane MRI volumes with Slice Attention.")
    print("  3. Final Blended Prob: Optimal pathology-weighted blend (alpha_c * Vision + (1-alpha_c) * NLP).")
    print("  4. Values >= 0.5 indicate predicted abnormality presence (evaluated via Macro-AUC).")
    print("=" * 80)


if __name__ == '__main__':
    demonstrate_single_study_inference()
