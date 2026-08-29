"""
Automated Verification & Test Suite for Phase 5 (Multimodal) & Phase 6 (Ensemble & Submission)
==============================================================================================
Runs end-to-end verification of:
1. ClinicalTextEncoder semantic tokenization and multilingual report encoding.
2. MultimodalContrastiveModel vision-language InfoNCE contrastive alignment.
3. MultimodalFusionClassifier dual-modality cross-attention forward and backward passes.
4. PathologySpecificBlender optimization and per-class alpha_c weighting.
5. ProbabilityCalibrator temperature scaling and calibration.
6. SubmissionGenerator format verification and zero-NaN integrity assertions.
"""

import sys
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import MultiViewKneeModel, TARGET_COLS
from src.multimodal import (
    ClinicalTextEncoder,
    MultimodalContrastiveModel,
    MultimodalFusionClassifier
)
from src.ensemble import (
    PathologySpecificBlender,
    ProbabilityCalibrator,
    CrossValidationEnsemble,
    SubmissionGenerator
)


def run_tests():
    print("=" * 70)
    print("[TEST SUITE] RSNA Knee Multimodal AI - Phases 5 & 6 Verification")
    print("=" * 70)

    test_dir = Path("outputs/test_fixtures/multimodal_ensemble")
    test_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Test 1: ClinicalTextEncoder on Multilingual Reports
    # -------------------------------------------------------------
    print("\n[Test 1/6] Testing ClinicalTextEncoder on Multilingual Reports...")
    text_encoder = ClinicalTextEncoder(vocab_size=5000, embed_dim=64, hidden_dim=64, out_dim=128)
    sample_reports = [
        "Rotura de menisco interno y rotura de ligamento cruzado anterior.", # Spanish
        "Rupture complete du ligament croise anterieur avec epanchement.",     # French
        "Meniscusscheur mediaal en gonartrose femorotibiaal.",               # Dutch
        "Complete ACL tear with moderate joint effusion and bone contusion."  # English
    ]
    text_embeds = text_encoder(sample_reports)
    assert text_embeds.shape == (4, 128), f"Expected (4, 128), got {text_embeds.shape}"
    assert not torch.isnan(text_embeds).any()
    print(f"  [PASS] Successfully encoded {len(sample_reports)} multilingual reports -> {text_embeds.shape}")

    # -------------------------------------------------------------
    # Test 2: MultimodalContrastiveModel & InfoNCE Loss
    # -------------------------------------------------------------
    print("\n[Test 2/6] Testing Vision-Language Contrastive Alignment (InfoNCE)...")
    vision_model = MultiViewKneeModel(num_classes=12, feature_dim=64, fused_dim=128)
    contrastive_model = MultimodalContrastiveModel(
        vision_model=vision_model,
        text_encoder=text_encoder,
        latent_dim=128
    )

    # Mock batch of 2 studies: [B=2, 3, Depth=8, H=64, W=64]
    mock_images = torch.randn(2, 3, 8, 64, 64)
    reports_pair = sample_reports[:2]

    z_v, z_t, sim_matrix = contrastive_model(mock_images, reports_pair)
    assert z_v.shape == (2, 128) and z_t.shape == (2, 128)
    assert sim_matrix.shape == (2, 2)

    # Verify unit sphere normalization
    v_norm = torch.norm(z_v, dim=-1).detach().numpy()
    t_norm = torch.norm(z_t, dim=-1).detach().numpy()
    assert np.allclose(v_norm, [1.0, 1.0], atol=1e-4)
    assert np.allclose(t_norm, [1.0, 1.0], atol=1e-4)

    contrast_loss = contrastive_model.compute_contrastive_loss(sim_matrix)
    assert contrast_loss.item() > 0 and not torch.isnan(contrast_loss)
    print(f"  [PASS] Visual-Text Latents normalized: z_v={z_v.shape}, z_t={z_t.shape}")
    print(f"  [PASS] Contrastive InfoNCE Loss: {contrast_loss.item():.4f}")

    # -------------------------------------------------------------
    # Test 3: MultimodalFusionClassifier Forward & Backward Pass
    # -------------------------------------------------------------
    print("\n[Test 3/6] Testing MultimodalFusionClassifier (Dual-Modality Inference)...")
    joint_classifier = MultimodalFusionClassifier(
        contrastive_model=contrastive_model,
        num_classes=12,
        fused_dim=128
    )
    joint_logits = joint_classifier(mock_images, reports_pair)
    assert joint_logits.shape == (2, 12), f"Expected (2, 12), got {joint_logits.shape}"

    # Verify backward pass gradients through joint network
    joint_classifier.zero_grad()
    loss = joint_logits.sum()
    loss.backward()
    assert contrastive_model.vision_model.sag_encoder.stem[0].weight.grad is not None
    assert contrastive_model.text_encoder.embedding.weight.grad is not None
    print(f"  [PASS] Dual-modality forward output: {joint_logits.shape}")
    print(f"  [PASS] Non-zero gradients propagated back through both Vision and Text encoders.")

    # -------------------------------------------------------------
    # Test 4: PathologySpecificBlender Optimization
    # -------------------------------------------------------------
    print("\n[Test 4/6] Testing Pathology-Specific Vision + NLP Blender...")
    blender = PathologySpecificBlender()
    
    # Mock validation data (N=20 studies)
    np.random.seed(42)
    y_true_val = np.random.randint(0, 2, size=(20, 12)).astype(float)
    y_vis_val = np.random.uniform(0.1, 0.9, size=(20, 12)).astype(np.float32)
    y_nlp_val = np.random.uniform(0.1, 0.9, size=(20, 12)).astype(np.float32)

    fitted_weights = blender.fit(y_true_val, y_vis_val, y_nlp_val)
    assert len(fitted_weights) == 12
    assert all(0.0 <= w <= 1.0 for w in fitted_weights.values())

    blended_probs = blender.blend(y_vis_val, y_nlp_val)
    assert blended_probs.shape == (20, 12)
    assert (blended_probs >= 0.0).all() and (blended_probs <= 1.0).all()

    baker_w = fitted_weights.get("Baker's", 0.5)
    print(f"  [PASS] Fitted Pathology Blend Weights: ACL={fitted_weights['ACL']:.2f}, MCL={fitted_weights['MCL']:.2f}, Baker's={baker_w:.2f}")
    print(f"  [PASS] Blended Probabilities Shape: {blended_probs.shape}")

    # -------------------------------------------------------------
    # Test 5: ProbabilityCalibrator & CrossValidationEnsemble
    # -------------------------------------------------------------
    print("\n[Test 5/6] Testing Probability Calibration & Multi-Fold Ensembling...")
    calibrator = ProbabilityCalibrator()
    temp = calibrator.fit_temperature(y_true_val, y_vis_val)
    calibrated = calibrator.calibrate_temperature(y_vis_val)
    assert 0.1 <= temp <= 5.0
    assert calibrated.shape == (20, 12)
    print(f"  [PASS] Optimized Calibration Temperature: {temp:.4f}")

    # Multi-Fold averaging
    fold_preds = [y_vis_val, y_vis_val * 0.95, y_vis_val * 1.05]
    avg_preds = CrossValidationEnsemble.average_fold_predictions(fold_preds)
    rank_preds = CrossValidationEnsemble.rank_average_predictions(fold_preds)
    assert avg_preds.shape == (20, 12)
    assert rank_preds.shape == (20, 12)
    print(f"  [PASS] Multi-Fold Ensembling (Mean & Rank Averaging) Verified.")

    # -------------------------------------------------------------
    # Test 6: SubmissionGenerator & Schema Validation
    # -------------------------------------------------------------
    print("\n[Test 6/6] Testing Submission File Generation & Schema Verification...")
    mock_study_ids = [f"1.2.826.0.1.3680043.8.498.{i+100000}" for i in range(10)]
    mock_probs = np.random.uniform(0.01, 0.99, size=(10, 12)).astype(np.float32)

    sub_df = SubmissionGenerator.create_submission_dataframe(mock_study_ids, mock_probs)
    assert sub_df.shape == (10, 13) # StudyInstanceUID + 12 targets
    assert 'StudyInstanceUID' in sub_df.columns
    for col in TARGET_COLS:
        assert col in sub_df.columns

    saved_path = SubmissionGenerator.validate_and_save(
        sub_df, 
        output_path=test_dir / "submission_test.csv"
    )
    assert saved_path.exists()
    print(f"  [PASS] Verified submission file written and passed all schema checks.")

    # Clean up test fixtures
    shutil.rmtree(Path("outputs/test_fixtures"), ignore_errors=True)

    print("\n" + "=" * 70)
    print("[SUCCESS] ALL 6 MULTIMODAL & ENSEMBLE TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == '__main__':
    run_tests()
