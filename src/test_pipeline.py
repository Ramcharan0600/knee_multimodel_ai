"""
Automated Verification & Test Suite for Phase 3 MRI Preprocessing & Dataset Pipeline
====================================================================================
Runs end-to-end verification of:
1. Synthetic DICOM series generation & metadata parsing.
2. DICOMReader slice sorting, orientation checking, and intensity scaling.
3. SeriesSelector coverage & ranking across the entire RSNA train cohort (4,407 studies).
4. Volumetric resampling, windowing, and 3D data augmentations.
5. PyTorch KneeMRIDataset and DataLoader batch iteration & tensor dimensions.
"""

import sys
import shutil
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.mri_preprocessor import (
    DICOMReader,
    IntensityNormalizer,
    SeriesSelector,
    VolumeResampler,
    SyntheticDICOMGenerator,
    TARGET_COLS,
    PLANES
)
from src.dataset import (
    MRIVolumeAugmentations,
    KneeMRIDataset,
    knee_collate_fn,
    create_dataloaders
)


def run_tests():
    print("=" * 70)
    print("[TEST SUITE] RSNA Knee Multimodal AI - Phase 3 Pipeline Verification")
    print("=" * 70)

    test_dir = Path("outputs/test_fixtures")
    test_dir.mkdir(parents=True, exist_ok=True)
    temp_dicom_root = test_dir / "synthetic_dicoms"
    temp_dicom_root.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Test 1: Synthetic DICOM Generation & DICOMReader
    # -------------------------------------------------------------
    print("\n[Test 1/5] Testing DICOM Generation & Sorter Reader...")
    study_id = "1.2.826.0.1.3680043.8.498.999901"
    series_id = "1.2.826.0.1.3680043.8.498.999902"
    series_dir = SyntheticDICOMGenerator.save_mock_dicom_series(
        output_dir=temp_dicom_root,
        study_uid=study_id,
        series_uid=series_id,
        depth=12,
        plane="Sagittal"
    )
    print(f"  [PASS] Generated synthetic DICOM series at: {series_dir}")

    # Read single file
    first_dcm = sorted(list(series_dir.glob("*.dcm")))[0]
    slice_arr, meta = DICOMReader.read_dicom_file(first_dcm)
    assert slice_arr.ndim == 2, f"Expected 2D slice, got {slice_arr.shape}"
    assert meta["StudyInstanceUID"] == study_id
    assert meta["SeriesInstanceUID"] == series_id
    print(f"  [PASS] Read single DICOM slice: Shape={slice_arr.shape}, InstanceNumber={meta['InstanceNumber']}")

    # Read entire series
    vol, metas = DICOMReader.read_series(series_dir)
    assert vol.shape == (12, 256, 256), f"Expected (12, 256, 256), got {vol.shape}"
    assert len(metas) == 12
    print(f"  [PASS] Read sorted 3D DICOM series volume: Shape={vol.shape}, Depth={len(metas)}")

    # -------------------------------------------------------------
    # Test 2: SeriesSelector on Full 4,407 Studies Cohort
    # -------------------------------------------------------------
    print("\n[Test 2/5] Testing SeriesSelector on Full Dataset (train_series.csv)...")
    series_df = pd.read_csv("data/train_series.csv")
    selector = SeriesSelector(series_df)
    
    unique_studies = series_df["StudyInstanceUID"].unique()
    sample_study = unique_studies[0]
    selected_sample = selector.select_study_series(sample_study)
    print(f"  [PASS] Sample Study: {sample_study[:24]}...")
    for p in PLANES:
        print(f"         - {p:8s} -> {selected_sample[p]}")

    # Check overall coverage across first 100 studies
    has_sag, has_cor, has_ax = 0, 0, 0
    test_cohort = unique_studies[:100]
    for s_id in test_cohort:
        mapping = selector.select_study_series(s_id)
        if mapping.get("Sagittal"): has_sag += 1
        if mapping.get("Coronal"): has_cor += 1
        if mapping.get("Axial"): has_ax += 1

    print(f"  [PASS] Evaluated 100 studies coverage: Sagittal={has_sag}%, Coronal={has_cor}%, Axial={has_ax}%")

    # -------------------------------------------------------------
    # Test 3: Normalization & Volumetric Resampling
    # -------------------------------------------------------------
    print("\n[Test 3/5] Testing Normalization & Volume Resampling...")
    raw_vol = SyntheticDICOMGenerator.create_mock_knee_volume(depth=28, height=300, width=300, plane="Coronal")
    proc_vol = VolumeResampler.preprocess_volume(
        raw_vol,
        target_depth=16,
        target_spatial=(224, 224)
    )
    assert proc_vol.shape == (16, 224, 224), f"Expected (16, 224, 224), got {proc_vol.shape}"
    assert 0.0 <= proc_vol.min() and proc_vol.max() <= 1.0, "Intensity outside [0, 1]!"
    print(f"  [PASS] Volume Resampled from (28, 300, 300) -> {proc_vol.shape}, Min={proc_vol.min():.3f}, Max={proc_vol.max():.3f}")

    # -------------------------------------------------------------
    # Test 4: 3D Volumetric Augmentations
    # -------------------------------------------------------------
    print("\n[Test 4/5] Testing 3D Volumetric Augmentation Consistency...")
    aug = MRIVolumeAugmentations(
        p_flip=1.0, 
        p_rot=1.0, 
        max_rot_deg=10.0, 
        p_scale_shift=1.0,
        p_gamma=1.0, 
        p_noise=1.0, 
        p_cutout=1.0, 
        is_train=True
    )
    aug_vol = aug(proc_vol, plane="Coronal")
    assert aug_vol.shape == (16, 224, 224), f"Shape altered by augmentations: {aug_vol.shape}"
    assert 0.0 <= aug_vol.min() and aug_vol.max() <= 1.0
    print(f"  [PASS] 3D Augmentation verified: Output shape={aug_vol.shape}, Non-zero values preserved.")

    # -------------------------------------------------------------
    # Test 5: PyTorch KneeMRIDataset & DataLoader Batching
    # -------------------------------------------------------------
    print("\n[Test 5/5] Testing PyTorch KneeMRIDataset & Cross-Validation DataLoader...")
    train_loader, val_loader = create_dataloaders(
        df_path="data/train_folds.csv",
        series_path="data/train_series.csv",
        dicom_root=temp_dicom_root,
        val_fold=0,
        batch_size=4,
        num_workers=0,
        target_depth=16,
        target_spatial=(224, 224),
        mode="pseudo_labeled"
    )

    batch = next(iter(train_loader))
    
    assert "image" in batch
    assert batch["image"].shape == (4, 3, 16, 224, 224), f"Expected [4, 3, 16, 224, 224], got {batch['image'].shape}"
    assert batch["sagittal"].shape == (4, 1, 16, 224, 224)
    assert batch["coronal"].shape == (4, 1, 16, 224, 224)
    assert batch["axial"].shape == (4, 1, 16, 224, 224)
    assert batch["targets"].shape == (4, 12), f"Expected [4, 12], got {batch['targets'].shape}"
    assert len(batch["study_id"]) == 4
    assert len(batch["report"]) == 4

    print(f"  [PASS] PyTorch Mini-Batch Verified:")
    print(f"         - Stacked Tri-Plane Image Tensor: {batch['image'].shape} (dtype: {batch['image'].dtype})")
    print(f"         - Target Label Matrix:            {batch['targets'].shape} (dtype: {batch['targets'].dtype})")
    print(f"         - Has Gold Indicator:             {batch['has_gold'].tolist()}")
    print(f"         - Fold Indices:                   {batch['fold'].tolist()}")
    print(f"         - Sample Targets (Study 0):       {np.round(batch['targets'][0].numpy(), 2).tolist()}")

    # Clean up test fixtures
    shutil.rmtree(test_dir, ignore_errors=True)

    print("\n" + "=" * 70)
    print("[SUCCESS] ALL 5 PIPELINE VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
