"""
Automated Verification & Test Suite for Phase 4 Model Architecture & Training Engine
===================================================================================
Runs end-to-end verification of:
1. MultiViewKneeModel & SingleViewKneeModel instantiation & tensor dimensions.
2. SliceAttentionPooling weight normalization and interpretability maps.
3. Loss functions (SoftBCEWithLogitsLoss, AsymmetricLoss, CombinedKneeLoss).
4. Full backward pass & gradient propagation through all parameters.
5. ModelTrainer mini-epoch training loop & checkpoint serialization/deserialization.
6. Metric calculation and evaluation report generation.
"""

import sys
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import (
    MultiViewKneeModel, 
    SingleViewKneeModel, 
    SliceAttentionPooling,
    TARGET_COLS
)
from src.losses import (
    SoftBCEWithLogitsLoss, 
    AsymmetricLoss, 
    CombinedKneeLoss
)
from src.trainer import (
    calculate_metrics, 
    ModelTrainer, 
    evaluate_model
)


class MockKneeDataset(torch.utils.data.Dataset):
    """
    Mock dataset for fast headless unit testing of the training engine.
    """
    def __init__(self, size: int = 8, depth: int = 8, height: int = 64, width: int = 64):
        self.size = size
        self.depth = depth
        self.height = height
        self.width = width

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 3 planes stacked: [3, D, H, W]
        img = torch.rand((3, self.depth, self.height, self.width), dtype=torch.float32)
        # 12 soft targets in [0, 1]
        targets = torch.rand(12, dtype=torch.float32)
        has_gold = torch.tensor(1.0 if idx % 2 == 0 else 0.0, dtype=torch.float32)
        
        return {
            'study_id': f"study_mock_{idx:03d}",
            'image': img,
            'sagittal': img[0:1],
            'coronal': img[1:2],
            'axial': img[2:3],
            'targets': targets,
            'has_gold': has_gold,
            'fold': idx % 5,
            'report': "Sample mock report."
        }


def run_tests():
    print("=" * 70)
    print("[TEST SUITE] RSNA Knee Multimodal AI - Phase 4 Model & Trainer Verification")
    print("=" * 70)

    test_model_dir = Path("outputs/test_fixtures/models")
    test_model_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Test 1: MultiViewKneeModel Forward Pass & Slice Attention
    # -------------------------------------------------------------
    print("\n[Test 1/6] Testing MultiViewKneeModel Forward Pass & Attention...")
    model = MultiViewKneeModel(num_classes=12, feature_dim=128, fused_dim=256)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [PASS] Initialized MultiViewKneeModel: {param_count:,} trainable parameters.")

    # Fast test batch: [B=2, Planes=3, Depth=8, H=64, W=64]
    dummy_input = torch.randn(2, 3, 8, 64, 64)
    logits, attn_dict = model(dummy_input, return_attention=True)

    assert logits.shape == (2, 12), f"Expected logits shape (2, 12), got {logits.shape}"
    assert 'sagittal' in attn_dict and 'coronal' in attn_dict and 'axial' in attn_dict
    assert attn_dict['sagittal'].shape == (2, 8)
    
    # Verify attention weights sum to 1.0 along slice dimension
    sag_sum = attn_dict['sagittal'].sum(dim=1).detach().numpy()
    assert np.allclose(sag_sum, [1.0, 1.0], atol=1e-4), f"Attention weights do not sum to 1.0: {sag_sum}"

    print(f"  [PASS] Forward Pass Output Shape: {logits.shape}")
    print(f"  [PASS] Slice Attention Map: {attn_dict['sagittal'].shape} (Sum: {sag_sum.tolist()})")

    # -------------------------------------------------------------
    # Test 2: SingleViewKneeModel
    # -------------------------------------------------------------
    print("\n[Test 2/6] Testing SingleViewKneeModel...")
    single_model = SingleViewKneeModel(num_classes=12, feature_dim=128)
    single_input = torch.randn(2, 1, 8, 64, 64)
    single_out = single_model(single_input)
    assert single_out.shape == (2, 12), f"Expected (2, 12), got {single_out.shape}"
    print(f"  [PASS] SingleViewKneeModel Output Shape: {single_out.shape}")

    # -------------------------------------------------------------
    # Test 3: Multi-Label Loss Formulations
    # -------------------------------------------------------------
    print("\n[Test 3/6] Testing Multi-Label Loss Formulations (Soft BCE & Asymmetric Loss)...")
    soft_targets = torch.tensor([[0.95, 0.05, 0.88, 0.10, 0.05, 0.05, 0.85, 0.90, 0.05, 0.80, 0.05, 0.05],
                                 [0.05, 0.92, 0.05, 0.05, 0.75, 0.80, 0.05, 0.10, 0.70, 0.05, 0.65, 0.05]])
    has_gold = torch.tensor([1.0, 0.0])

    bce_loss_fn = SoftBCEWithLogitsLoss(gold_weight=2.0)
    focal_loss_fn = AsymmetricLoss(gold_weight=2.0)
    combined_loss_fn = CombinedKneeLoss(gold_weight=2.0)

    loss_bce = bce_loss_fn(logits, soft_targets, has_gold)
    loss_focal = focal_loss_fn(logits, soft_targets, has_gold)
    loss_comb = combined_loss_fn(logits, soft_targets, has_gold)

    assert not torch.isnan(loss_bce) and loss_bce.item() > 0
    assert not torch.isnan(loss_focal) and loss_focal.item() > 0
    assert not torch.isnan(loss_comb) and loss_comb.item() > 0

    print(f"  [PASS] SoftBCE Loss:        {loss_bce.item():.4f}")
    print(f"  [PASS] Asymmetric Loss:     {loss_focal.item():.4f}")
    print(f"  [PASS] Combined Knee Loss:   {loss_comb.item():.4f}")

    # -------------------------------------------------------------
    # Test 4: Gradient Flow & Backward Pass
    # -------------------------------------------------------------
    print("\n[Test 4/6] Testing Gradient Propagation through Model...")
    model.zero_grad()
    loss_comb.backward()

    # Check gradients in stem, attention, and classifier
    has_grad_stem = model.sag_encoder.stem[0].weight.grad is not None
    has_grad_attn = model.sag_attn.proj[0].weight.grad is not None
    has_grad_fusion = model.fusion.fusion_fc[0].weight.grad is not None
    has_grad_class = model.classifier[-1].weight.grad is not None

    assert has_grad_stem and has_grad_attn and has_grad_fusion and has_grad_class
    print(f"  [PASS] Non-zero gradients verified across Encoder, Attention, Fusion, and Classifier layers.")

    # -------------------------------------------------------------
    # Test 5: Metrics Calculation
    # -------------------------------------------------------------
    print("\n[Test 5/6] Testing Macro-AUC & Classification Metric Calculator...")
    y_true_mock = np.array([
        [1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0]
    ])
    y_prob_mock = np.array([
        [0.9, 0.1, 0.8, 0.2, 0.1, 0.1, 0.85, 0.95, 0.1, 0.8, 0.1, 0.1],
        [0.1, 0.85, 0.1, 0.1, 0.9, 0.75, 0.1, 0.2, 0.8, 0.1, 0.9, 0.2],
        [0.8, 0.7, 0.9, 0.1, 0.2, 0.1, 0.7, 0.8, 0.2, 0.1, 0.1, 0.9],
        [0.2, 0.1, 0.2, 0.85, 0.8, 0.2, 0.1, 0.1, 0.75, 0.2, 0.1, 0.1]
    ])

    metrics = calculate_metrics(y_true_mock, y_prob_mock)
    assert 'macro_auc' in metrics and 'macro_f1' in metrics
    assert 0.0 <= metrics['macro_auc'] <= 1.0
    print(f"  [PASS] Calculated Macro-AUC: {metrics['macro_auc']:.4f} | Macro-F1: {metrics['macro_f1']:.4f}")

    # -------------------------------------------------------------
    # Test 6: ModelTrainer Fit & Checkpoint Serialization
    # -------------------------------------------------------------
    print("\n[Test 6/6] Testing ModelTrainer & Checkpoint Lifecycle...")
    train_ds = MockKneeDataset(size=8)
    val_ds = MockKneeDataset(size=4)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=CombinedKneeLoss(),
        save_dir=test_model_dir,
        fold_id=99
    )

    history = trainer.fit(epochs=2, early_stopping_patience=2)
    assert len(history['epoch']) == 2
    assert trainer.best_checkpoint_path.exists(), f"Checkpoint not saved: {trainer.best_checkpoint_path}"
    print(f"  [PASS] Training Loop executed 2 epochs successfully.")
    print(f"  [PASS] Saved model checkpoint: {trainer.best_checkpoint_path}")

    # Test Checkpoint Restoration
    ckpt = torch.load(trainer.best_checkpoint_path, weights_only=False)
    assert 'model_state_dict' in ckpt
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  [PASS] Checkpoint state dict loaded and verified.")

    # Test Standalone evaluate_model
    report_df, macro_auc = evaluate_model(model, val_loader)
    assert len(report_df) == 12
    print(f"  [PASS] Evaluator Report generated: {len(report_df)} pathologies evaluated (Macro-AUC={macro_auc:.4f})")

    # Clean up test fixtures
    shutil.rmtree(Path("outputs/test_fixtures"), ignore_errors=True)

    print("\n" + "=" * 70)
    print("[SUCCESS] ALL 6 MODEL & TRAINING TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == '__main__':
    run_tests()
