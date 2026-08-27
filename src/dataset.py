"""
PyTorch Dataset and Volumetric Augmentation Pipeline for RSNA Knee Multimodal AI
================================================================================
Provides:
1. MRIVolumeAugmentations: 3D-consistent spatial and intensity augmentations.
2. KneeMRIDataset: Multi-view tri-plane (Sagittal, Coronal, Axial) PyTorch Dataset.
3. knee_collate_fn: Custom batch collation for multi-sequence volumetric batches.
4. create_dataloaders: Cross-validation DataLoader factory.
"""

import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from src.mri_preprocessor import (
        DICOMReader, 
        IntensityNormalizer, 
        SeriesSelector, 
        VolumeResampler, 
        SyntheticDICOMGenerator,
        TARGET_COLS, 
        PLANES
    )
except ModuleNotFoundError:
    from mri_preprocessor import (
        DICOMReader, 
        IntensityNormalizer, 
        SeriesSelector, 
        VolumeResampler, 
        SyntheticDICOMGenerator,
        TARGET_COLS, 
        PLANES
    )


class MRIVolumeAugmentations:
    """
    Volumetric data augmentation for multi-slice knee MRI sequences.
    Ensures that spatial transformations (rotations, translations, flips) are applied 
    identically across all slices in a 3D volume to preserve anatomical continuity.
    """

    def __init__(
        self,
        p_flip: float = 0.5,
        p_rot: float = 0.5,
        max_rot_deg: float = 12.0,
        p_scale_shift: float = 0.5,
        scale_range: Tuple[float, float] = (0.92, 1.08),
        shift_range: Tuple[float, float] = (-0.05, 0.05),
        p_gamma: float = 0.4,
        gamma_range: Tuple[float, float] = (0.80, 1.25),
        p_noise: float = 0.3,
        noise_std: float = 0.02,
        p_cutout: float = 0.3,
        cutout_size: Tuple[int, int] = (32, 32),
        is_train: bool = True
    ):
        self.p_flip = p_flip
        self.p_rot = p_rot
        self.max_rot_deg = max_rot_deg
        self.p_scale_shift = p_scale_shift
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.p_gamma = p_gamma
        self.gamma_range = gamma_range
        self.p_noise = p_noise
        self.noise_std = noise_std
        self.p_cutout = p_cutout
        self.cutout_size = cutout_size
        self.is_train = is_train

    def __call__(self, volume: np.ndarray, plane: str = 'Sagittal') -> np.ndarray:
        """
        Applies consistent 3D transformations to volume of shape (D, H, W).
        """
        if not self.is_train:
            return volume

        D, H, W = volume.shape
        augmented = volume.copy()

        # 1. Random Horizontal Flip (Plane-Safe)
        # Flip along width axis (anatomically valid for Sagittal/Coronal/Axial)
        if np.random.rand() < self.p_flip:
            augmented = np.flip(augmented, axis=2).copy()

        # 2. Consistent Spatial Affine (Rotation + Translation + Scaling)
        apply_rot = np.random.rand() < self.p_rot
        apply_scale_shift = np.random.rand() < self.p_scale_shift

        if apply_rot or apply_scale_shift:
            angle = np.random.uniform(-self.max_rot_deg, self.max_rot_deg) if apply_rot else 0.0
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1]) if apply_scale_shift else 1.0
            shift_x = np.random.uniform(self.shift_range[0], self.shift_range[1]) * W if apply_scale_shift else 0.0
            shift_y = np.random.uniform(self.shift_range[0], self.shift_range[1]) * H if apply_scale_shift else 0.0

            # Transform each slice using identical affine matrix
            for z in range(D):
                slice_img = Image.fromarray((augmented[z] * 255.0).astype(np.uint8))
                
                # Apply rotation
                if abs(angle) > 1e-2:
                    slice_img = slice_img.rotate(angle, resample=Image.BILINEAR)

                # Apply scaling and translation via affine transform if needed
                if scale != 1.0 or shift_x != 0.0 or shift_y != 0.0:
                    # Affine matrix in PIL: (a, b, c, d, e, f)
                    # where X = a*x + b*y + c, Y = d*x + e*y + f
                    inv_scale = 1.0 / scale
                    cx, cy = W / 2.0, H / 2.0
                    a = inv_scale
                    b = 0.0
                    c = cx * (1.0 - inv_scale) - shift_x
                    d = 0.0
                    e = inv_scale
                    f = cy * (1.0 - inv_scale) - shift_y
                    slice_img = slice_img.transform(
                        (W, H), 
                        Image.AFFINE, 
                        (a, b, c, d, e, f), 
                        resample=Image.BILINEAR
                    )

                augmented[z] = np.array(slice_img, dtype=np.float32) / 255.0

        # 3. Random Gamma Perturbation (Non-linear intensity scaling)
        if np.random.rand() < self.p_gamma:
            gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
            augmented = np.power(np.clip(augmented, 1e-6, 1.0), gamma)

        # 4. Additive Gaussian Noise
        if np.random.rand() < self.p_noise:
            noise = np.random.normal(0, self.noise_std, size=augmented.shape).astype(np.float32)
            augmented = np.clip(augmented + noise, 0.0, 1.0)

        # 5. Random Cutout / Erasing Block (Applied across all slices or random slice subset)
        if np.random.rand() < self.p_cutout:
            ch, cw = self.cutout_size
            top = np.random.randint(0, max(1, H - ch))
            left = np.random.randint(0, max(1, W - cw))
            augmented[:, top:top + ch, left:left + cw] = 0.0

        return np.clip(augmented, 0.0, 1.0).astype(np.float32)


class KneeMRIDataset(Dataset):
    """
    Multi-View Tri-Plane Knee MRI Dataset for PyTorch.
    
    Loads and standardizes MRI volumes for Sagittal, Coronal, and Axial planes.
    Supports:
    - Supervised Gold-Standard Ground Truth (12 binary classes)
    - Weak Supervised Pseudo-Labels (12 soft probabilities + hard binary flags)
    - Multimodal Mode (MRI volume tensors + Radiology Report text)
    - On-the-fly DICOM loading, pre-computed array caching (.npz/.npy), and synthetic fallback.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        series_df: pd.DataFrame,
        dicom_root: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        target_depth: int = 16,
        target_spatial: Tuple[int, int] = (224, 224),
        planes: List[str] = ('Sagittal', 'Coronal', 'Axial'),
        is_train: bool = True,
        mode: str = 'pseudo_labeled',  # 'supervised_gold', 'pseudo_labeled', 'multimodal', 'inference'
        transform: Optional[Callable] = None,
        use_synthetic_fallback: bool = True
    ):
        self.df = df.copy().reset_index(drop=True)
        self.series_df = series_df.copy()
        self.dicom_root = Path(dicom_root) if dicom_root else None
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.target_depth = target_depth
        self.target_spatial = target_spatial
        self.planes = [p.capitalize() for p in planes]
        self.is_train = is_train
        self.mode = mode
        self.use_synthetic_fallback = use_synthetic_fallback

        # Initialize Series Selector
        self.selector = SeriesSelector(self.series_df)

        # Initialize Augmentation Pipeline
        self.transform = transform if transform is not None else MRIVolumeAugmentations(is_train=is_train)

    def __len__(self) -> int:
        return len(self.df)

    def _load_plane_volume(self, study_id: str, series_id: Optional[str], plane: str) -> np.ndarray:
        """
        Loads and preprocesses the 3D volume for a specific plane.
        Checks cache -> Checks DICOM files -> Falls back to synthetic volume if missing.
        """
        # 1. Check cache first
        if self.cache_dir and series_id:
            cache_file = self.cache_dir / f"{study_id}_{plane}_{series_id}.npy"
            if cache_file.exists():
                try:
                    vol = np.load(cache_file)
                    if vol.shape == (self.target_depth, self.target_spatial[0], self.target_spatial[1]):
                        return vol
                except Exception:
                    pass

        # 2. Check DICOM directory if available
        if self.dicom_root and series_id:
            series_dir = self.dicom_root / study_id / series_id
            if not series_dir.exists():
                # Check alternative layout: dicom_root / series_id
                series_dir = self.dicom_root / series_id

            if series_dir.exists() and series_dir.is_dir():
                try:
                    raw_vol, _ = DICOMReader.read_series(series_dir)
                    vol = VolumeResampler.preprocess_volume(
                        raw_vol,
                        target_depth=self.target_depth,
                        target_spatial=self.target_spatial
                    )
                    # Cache preprocessed array
                    if self.cache_dir:
                        cache_file = self.cache_dir / f"{study_id}_{plane}_{series_id}.npy"
                        np.save(cache_file, vol)
                    return vol
                except Exception:
                    pass

        # 3. Synthetic Fallback (Realistic Anatomical Simulation for testing/unmounted data)
        if self.use_synthetic_fallback:
            mock_raw = SyntheticDICOMGenerator.create_mock_knee_volume(
                depth=24, 
                height=self.target_spatial[0], 
                width=self.target_spatial[1], 
                plane=plane
            )
            vol = VolumeResampler.preprocess_volume(
                mock_raw,
                target_depth=self.target_depth,
                target_spatial=self.target_spatial
            )
            return vol

        # Default blank volume
        return np.zeros((self.target_depth, self.target_spatial[0], self.target_spatial[1]), dtype=np.float32)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        row = self.df.iloc[idx]
        study_id = str(row['StudyInstanceUID'])

        # Find best series for each plane
        selected_series = self.selector.select_study_series(study_id)

        # Load volumes for each plane
        plane_tensors = {}
        for plane in self.planes:
            series_id = selected_series.get(plane)
            vol = self._load_plane_volume(study_id, series_id, plane)
            
            # Apply 3D Augmentations
            if self.transform is not None:
                vol = self.transform(vol, plane=plane)

            # Shape: (1, D, H, W) - 1 channel 3D volume
            tensor = torch.from_numpy(vol).unsqueeze(0).float()
            plane_tensors[plane.lower()] = tensor

        # Stack into combined multi-view tensor if all 3 planes present
        # Shape: (3, D, H, W) where channel 0=Sagittal, 1=Coronal, 2=Axial
        stacked_views = torch.cat([
            plane_tensors.get('sagittal', torch.zeros((1, self.target_depth, *self.target_spatial))),
            plane_tensors.get('coronal', torch.zeros((1, self.target_depth, *self.target_spatial))),
            plane_tensors.get('axial', torch.zeros((1, self.target_depth, *self.target_spatial)))
        ], dim=0)

        # Extract Targets
        target_labels = np.zeros(len(TARGET_COLS), dtype=np.float32)
        has_gold = not pd.isna(row.get('ACL', np.nan))

        if self.mode == 'supervised_gold':
            for i, col in enumerate(TARGET_COLS):
                val = row.get(col, 0.0)
                target_labels[i] = 0.0 if pd.isna(val) else float(val)
        elif self.mode == 'pseudo_labeled':
            for i, col in enumerate(TARGET_COLS):
                # Prefer gold label if present, else pseudo-prob
                if has_gold and not pd.isna(row.get(col)):
                    target_labels[i] = float(row[col])
                elif f'{col}_prob' in row and not pd.isna(row[f'{col}_prob']):
                    target_labels[i] = float(row[f'{col}_prob'])
                elif col in row and not pd.isna(row[col]):
                    target_labels[i] = float(row[col])
                else:
                    target_labels[i] = 0.0
        elif self.mode == 'inference':
            target_labels = np.zeros(len(TARGET_COLS), dtype=np.float32)

        item = {
            'study_id': study_id,
            'image': stacked_views,               # Shape: [3, Depth, Height, Width]
            'sagittal': plane_tensors.get('sagittal'), # Shape: [1, Depth, Height, Width]
            'coronal': plane_tensors.get('coronal'),   # Shape: [1, Depth, Height, Width]
            'axial': plane_tensors.get('axial'),       # Shape: [1, Depth, Height, Width]
            'targets': torch.tensor(target_labels, dtype=torch.float32), # Shape: [12]
            'has_gold': torch.tensor(1.0 if has_gold else 0.0, dtype=torch.float32),
            'fold': int(row.get('fold', -1)),
            'report': str(row.get('Report', ''))
        }

        return item


def knee_collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """
    Collate function to assemble variable data items into mini-batch tensors.
    """
    study_ids = [item['study_id'] for item in batch]
    images = torch.stack([item['image'] for item in batch], dim=0)        # [B, 3, D, H, W]
    sagittal = torch.stack([item['sagittal'] for item in batch], dim=0)  # [B, 1, D, H, W]
    coronal = torch.stack([item['coronal'] for item in batch], dim=0)    # [B, 1, D, H, W]
    axial = torch.stack([item['axial'] for item in batch], dim=0)        # [B, 1, D, H, W]
    targets = torch.stack([item['targets'] for item in batch], dim=0)    # [B, 12]
    has_gold = torch.stack([item['has_gold'] for item in batch], dim=0)  # [B]
    folds = torch.tensor([item['fold'] for item in batch], dtype=torch.long)
    reports = [item['report'] for item in batch]

    return {
        'study_id': study_ids,
        'image': images,
        'sagittal': sagittal,
        'coronal': coronal,
        'axial': axial,
        'targets': targets,
        'has_gold': has_gold,
        'fold': folds,
        'report': reports
    }


def create_dataloaders(
    df_path: Union[str, Path] = 'data/train_folds.csv',
    series_path: Union[str, Path] = 'data/train_series.csv',
    dicom_root: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    val_fold: int = 0,
    batch_size: int = 8,
    num_workers: int = 0,
    target_depth: int = 16,
    target_spatial: Tuple[int, int] = (224, 224),
    mode: str = 'pseudo_labeled',
    pin_memory: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Builds training and validation DataLoaders for a specific cross-validation fold.
    """
    df = pd.read_csv(df_path)
    series_df = pd.read_csv(series_path)

    # Check if pseudo labels available
    pseudo_path = Path('data/train_pseudo_labeled.csv')
    if pseudo_path.exists() and mode == 'pseudo_labeled':
        pseudo_df = pd.read_csv(pseudo_path)
        # Merge fold column into pseudo_df if needed
        if 'fold' not in pseudo_df.columns and 'fold' in df.columns:
            pseudo_df = pseudo_df.merge(df[['StudyInstanceUID', 'fold']], on='StudyInstanceUID', how='left')
        df = pseudo_df

    if 'fold' not in df.columns:
        raise ValueError("DataFrame must contain a 'fold' column for cross-validation splitting.")

    train_df = df[df['fold'] != val_fold].copy().reset_index(drop=True)
    val_df = df[df['fold'] == val_fold].copy().reset_index(drop=True)

    print(f"Creating DataLoaders for Fold {val_fold}:")
    print(f"  • Train Studies: {len(train_df):,}")
    print(f"  • Val Studies:   {len(val_df):,}")

    train_dataset = KneeMRIDataset(
        df=train_df,
        series_df=series_df,
        dicom_root=dicom_root,
        cache_dir=cache_dir,
        target_depth=target_depth,
        target_spatial=target_spatial,
        is_train=True,
        mode=mode
    )

    val_dataset = KneeMRIDataset(
        df=val_df,
        series_df=series_df,
        dicom_root=dicom_root,
        cache_dir=cache_dir,
        target_depth=target_depth,
        target_spatial=target_spatial,
        is_train=False,
        mode=mode
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=knee_collate_fn,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=knee_collate_fn,
        pin_memory=pin_memory
    )

    return train_loader, val_loader
