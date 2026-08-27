"""
MRI Preprocessing Module for RSNA Knee Multimodal AI
====================================================
Handles:
1. DICOM file reading, metadata parsing, and spatial slice ordering.
2. Rescale slope/intercept adjustment & PhotometricInterpretation inversion.
3. Robust anatomical percentile-based intensity windowing & normalization.
4. Intelligent series selection & ranking for Sagittal, Coronal, and Axial planes.
5. Volumetric resampling to fixed depth (D) and spatial dimensions (H, W).
6. Synthetic DICOM volume generation for testing and offline development.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image
import pydicom
from pydicom.dataset import FileDataset


# Target pathologies
TARGET_COLS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

# Standard Anatomical Planes
PLANES = ['Sagittal', 'Coronal', 'Axial']


class DICOMReader:
    """
    Robust medical DICOM image and series reader.
    Handles single-frame DICOM slices, multi-frame series, slice ordering,
    intensity windowing, and photometric interpretation.
    """

    @staticmethod
    def read_dicom_file(file_path: Union[str, Path]) -> Tuple[np.ndarray, dict]:
        """
        Reads a single DICOM file, applies slope/intercept correction, 
        and extracts relevant spatial metadata.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"DICOM file not found: {file_path}")

        dcm = pydicom.dcmread(str(file_path), force=True)
        
        # Extract metadata
        meta = {
            'StudyInstanceUID': getattr(dcm, 'StudyInstanceUID', ''),
            'SeriesInstanceUID': getattr(dcm, 'SeriesInstanceUID', ''),
            'SOPInstanceUID': getattr(dcm, 'SOPInstanceUID', ''),
            'InstanceNumber': int(getattr(dcm, 'InstanceNumber', 0)),
            'SliceLocation': float(getattr(dcm, 'SliceLocation', 0.0)) if hasattr(dcm, 'SliceLocation') else None,
            'ImagePositionPatient': [float(x) for x in getattr(dcm, 'ImagePositionPatient', [0.0, 0.0, 0.0])],
            'ImageOrientationPatient': [float(x) for x in getattr(dcm, 'ImageOrientationPatient', [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])],
            'PixelSpacing': [float(x) for x in getattr(dcm, 'PixelSpacing', [1.0, 1.0])],
            'Rows': int(getattr(dcm, 'Rows', 0)),
            'Columns': int(getattr(dcm, 'Columns', 0)),
            'PhotometricInterpretation': getattr(dcm, 'PhotometricInterpretation', 'MONOCHROME2'),
            'RescaleSlope': float(getattr(dcm, 'RescaleSlope', 1.0)),
            'RescaleIntercept': float(getattr(dcm, 'RescaleIntercept', 0.0)),
            'WindowCenter': getattr(dcm, 'WindowCenter', None),
            'WindowWidth': getattr(dcm, 'WindowWidth', None),
        }

        # Extract raw pixel array
        pixel_array = dcm.pixel_array.astype(np.float32)

        # Apply Rescale Slope and Intercept if present
        slope = meta['RescaleSlope']
        intercept = meta['RescaleIntercept']
        if slope != 1.0 or intercept != 0.0:
            pixel_array = pixel_array * slope + intercept

        # Handle Photometric Interpretation (MONOCHROME1 is inverted: 0 = White)
        if meta['PhotometricInterpretation'] == 'MONOCHROME1':
            pixel_array = np.amax(pixel_array) - pixel_array

        return pixel_array, meta

    @staticmethod
    def read_series(series_dir: Union[str, Path]) -> Tuple[np.ndarray, List[dict]]:
        """
        Reads all DICOM files in a series directory, sorts slices along the anatomical 
        trajectory, and returns a 3D numpy volume of shape (D, H, W).
        """
        series_dir = Path(series_dir)
        if not series_dir.exists() or not series_dir.is_dir():
            raise FileNotFoundError(f"Series directory not found: {series_dir}")

        dcm_files = [f for f in series_dir.iterdir() if f.is_file() and not f.name.startswith('.')]
        if not dcm_files:
            raise ValueError(f"No DICOM files found in series directory: {series_dir}")

        slices = []
        for f in dcm_files:
            try:
                arr, meta = DICOMReader.read_dicom_file(f)
                slices.append((arr, meta))
            except Exception as e:
                continue

        if not slices:
            raise RuntimeError(f"Failed to load any valid DICOM files from {series_dir}")

        # Sort slices by (1) ImagePositionPatient plane normal projection, (2) SliceLocation, or (3) InstanceNumber
        def get_slice_sort_key(item):
            _, meta = item
            if meta['ImagePositionPatient'] and len(meta['ImagePositionPatient']) == 3:
                iop = meta['ImageOrientationPatient']
                if len(iop) == 6:
                    row_vec = np.array(iop[:3])
                    col_vec = np.array(iop[3:])
                    normal_vec = np.cross(row_vec, col_vec)
                    pos_vec = np.array(meta['ImagePositionPatient'])
                    return np.dot(pos_vec, normal_vec)
                return meta['ImagePositionPatient'][2]
            if meta['SliceLocation'] is not None:
                return meta['SliceLocation']
            return meta['InstanceNumber']

        slices.sort(key=get_slice_sort_key)

        volume = np.stack([s[0] for s in slices], axis=0)  # (D, H, W)
        metas = [s[1] for s in slices]

        return volume, metas


class IntensityNormalizer:
    """
    Standardizes MRI intensity distributions across different scanners and sequence protocols.
    Uses percentile-based dynamic windowing to clip background noise and extreme high-intensity artifacts.
    """

    @staticmethod
    def normalize_volume(
        volume: np.ndarray, 
        lower_percentile: float = 1.0, 
        upper_percentile: float = 99.0
    ) -> np.ndarray:
        """
        Normalizes a 2D slice or 3D volume to [0.0, 1.0] using percentile windowing.
        """
        volume = volume.astype(np.float32)
        
        # Exclude background zero/near-zero values when computing percentiles if possible
        non_zero = volume[volume > np.mean(volume) * 0.1]
        if len(non_zero) > 100:
            p_low = np.percentile(non_zero, lower_percentile)
            p_high = np.percentile(non_zero, upper_percentile)
        else:
            p_low = np.percentile(volume, lower_percentile)
            p_high = np.percentile(volume, upper_percentile)

        if p_high > p_low:
            clipped = np.clip(volume, p_low, p_high)
            normalized = (clipped - p_low) / (p_high - p_low)
        else:
            v_max = np.max(volume)
            v_min = np.min(volume)
            if v_max > v_min:
                normalized = (volume - v_min) / (v_max - v_min)
            else:
                normalized = np.zeros_like(volume)

        return np.clip(normalized, 0.0, 1.0).astype(np.float32)


class SeriesSelector:
    """
    Intelligently selects and ranks the best MRI series for each anatomical plane
    (Sagittal, Coronal, Axial) from the study's available sequences in train_series.csv.
    
    Sequence Quality Scoring:
        Score = 2 * Fluid_Sensitive + 1 * Fat_Suppression
    Fluid-sensitive, fat-suppressed sequences (e.g. T2/PD FS) are given highest priority
    as they offer maximal diagnostic clarity for ligament tears, meniscal pathology, effusion, and edema.
    """

    def __init__(self, series_df: pd.DataFrame):
        self.series_df = series_df.copy()
        # Compute priority score
        self.series_df['Priority_Score'] = (
            2 * self.series_df['Fluid_Sensitive'].fillna(0).astype(int) + 
            1 * self.series_df['Fat_Suppression'].fillna(0).astype(int)
        )

    def select_study_series(self, study_id: str) -> Dict[str, Optional[str]]:
        """
        Returns a dict mapping plane -> SeriesInstanceUID for the given study.
        {'Sagittal': '...', 'Coronal': '...', 'Axial': '...'}
        """
        study_records = self.series_df[self.series_df['StudyInstanceUID'] == study_id]
        selected = {'Sagittal': None, 'Coronal': None, 'Axial': None}

        if study_records.empty:
            return selected

        for plane in PLANES:
            plane_records = study_records[study_records['Anatomical_Plane'].str.lower() == plane.lower()]
            if not plane_records.empty:
                # Rank by priority score descending, then take top series
                best_row = plane_records.sort_values(by='Priority_Score', ascending=False).iloc[0]
                selected[plane] = best_row['SeriesInstanceUID']

        return selected

    def get_study_series_info(self, study_id: str) -> pd.DataFrame:
        """
        Returns full series metadata for a specific study.
        """
        return self.series_df[self.series_df['StudyInstanceUID'] == study_id]


class VolumeResampler:
    """
    Resamples 3D MRI volumes to standard tensor dimensions (Depth, Height, Width).
    Supports uniform linspace slice selection and bilinear spatial resizing.
    """

    @staticmethod
    def resample_depth(volume: np.ndarray, target_depth: int = 16) -> np.ndarray:
        """
        Resamples a volume of shape (D, H, W) along the depth axis to target_depth slices.
        Uses equidistant slice sampling.
        """
        current_depth = volume.shape[0]
        if current_depth == target_depth:
            return volume

        if current_depth < 1:
            raise ValueError(f"Volume has invalid depth: {current_depth}")

        # Choose equidistant indices
        indices = np.linspace(0, current_depth - 1, target_depth).round().astype(int)
        return volume[indices]

    @staticmethod
    def resize_spatial(
        volume: np.ndarray, 
        target_size: Tuple[int, int] = (224, 224),
        resample_mode: int = Image.BILINEAR
    ) -> np.ndarray:
        """
        Resizes each 2D slice in a volume of shape (D, H, W) to (D, target_size[0], target_size[1]).
        """
        D, H, W = volume.shape
        target_h, target_w = target_size
        if (H, W) == (target_h, target_w):
            return volume

        resized_slices = []
        for i in range(D):
            slice_img = Image.fromarray((volume[i] * 255.0).astype(np.uint8))
            resized_slice = slice_img.resize((target_w, target_h), resample=resample_mode)
            resized_slices.append(np.array(resized_slice, dtype=np.float32) / 255.0)

        return np.stack(resized_slices, axis=0)

    @classmethod
    def preprocess_volume(
        cls, 
        volume: np.ndarray, 
        target_depth: int = 16, 
        target_spatial: Tuple[int, int] = (224, 224)
    ) -> np.ndarray:
        """
        Complete preprocessing pipeline for a raw volume:
        1. Intensity normalization [0, 1]
        2. Depth resampling to target_depth
        3. Spatial resizing to target_spatial
        Returns volume of shape (target_depth, target_spatial[0], target_spatial[1])
        """
        norm_vol = IntensityNormalizer.normalize_volume(volume)
        depth_vol = cls.resample_depth(norm_vol, target_depth=target_depth)
        spatial_vol = cls.resize_spatial(depth_vol, target_size=target_spatial)
        return spatial_vol.astype(np.float32)


class SyntheticDICOMGenerator:
    """
    Utility to generate realistic synthetic knee MRI DICOM datasets and mock volumes
    for robust offline pipeline testing, continuous integration, and debugging.
    """

    @staticmethod
    def create_mock_knee_volume(
        depth: int = 24, 
        height: int = 256, 
        width: int = 256, 
        plane: str = 'Sagittal',
        add_pathology: bool = False
    ) -> np.ndarray:
        """
        Creates a synthetic 3D volume mimicking knee MRI anatomical structures
        (Femur, Tibia, Patella, Menisci, Ligaments, and background noise).
        """
        vol = np.zeros((depth, height, width), dtype=np.float32)
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height / 2, width / 2

        for z in range(depth):
            # Base elliptical soft-tissue outline
            dist_sq = ((x - center_x) / (width * 0.38))**2 + ((y - center_y) / (height * 0.42))**2
            mask = dist_sq <= 1.0
            vol[z][mask] = 0.35 + 0.05 * np.sin(z / 3.0)

            # Femoral condyles (Upper bone)
            femur_dist = ((x - center_x) / (width * 0.28))**2 + ((y - (center_y - height * 0.18)) / (height * 0.22))**2
            vol[z][femur_dist <= 1.0] = 0.85

            # Tibial plateau (Lower bone)
            tibia_dist = ((x - center_x) / (width * 0.30))**2 + ((y - (center_y + height * 0.22)) / (height * 0.20))**2
            vol[z][tibia_dist <= 1.0] = 0.80

            # Joint space & meniscal wedge
            joint_y = center_y + height * 0.02
            if abs(y - joint_y).min() < height * 0.06:
                # Meniscal structures
                meniscus_mask = (np.abs(y - joint_y) < height * 0.04) & (np.abs(x - center_x) > width * 0.12) & (dist_sq <= 0.85)
                vol[z][meniscus_mask] = 0.15

            # Ligament structure (e.g. ACL in sagittal mid-slices)
            if plane.lower() == 'sagittal' and abs(z - depth / 2) < 4:
                lig_mask = (np.abs((y - center_y) - 0.8 * (x - center_x)) < 6) & (np.abs(x - center_x) < width * 0.15)
                vol[z][lig_mask] = 0.10 if not add_pathology else 0.75  # High signal if tear/edema

            # Add gentle Gaussian noise
            noise = np.random.normal(0, 0.02, size=(height, width)).astype(np.float32)
            vol[z] = np.clip(vol[z] + noise, 0.0, 1.0)

        return vol

    @staticmethod
    def save_mock_dicom_series(
        output_dir: Union[str, Path], 
        study_uid: str, 
        series_uid: str, 
        depth: int = 16, 
        plane: str = 'Sagittal'
    ) -> Path:
        """
        Saves a synthetic DICOM series directory with realistic headers.
        """
        output_dir = Path(output_dir)
        series_dir = output_dir / study_uid / series_uid
        series_dir.mkdir(parents=True, exist_ok=True)

        vol = SyntheticDICOMGenerator.create_mock_knee_volume(depth=depth, plane=plane)
        
        # Ensure valid DICOM UID formats
        valid_study_uid = study_uid if study_uid.startswith('1.2.') else f"1.2.826.0.1.3680043.8.498.{abs(hash(study_uid)) % 10000000}"
        valid_series_uid = series_uid if series_uid.startswith('1.2.') else f"1.2.826.0.1.3680043.8.498.{abs(hash(series_uid)) % 10000000}"

        for z in range(depth):
            file_meta = pydicom.dataset.FileMetaDataset()
            file_meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
            file_meta.MediaStorageSOPInstanceUID = f"{valid_series_uid}.{z+1}"
            file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

            ds = FileDataset(str(series_dir / f"slice_{z:03d}.dcm"), {}, file_meta=file_meta, preamble=b"\0" * 128)
            ds.StudyInstanceUID = valid_study_uid
            ds.SeriesInstanceUID = valid_series_uid
            ds.SOPInstanceUID = f"{valid_series_uid}.{z+1}"
            ds.InstanceNumber = z + 1
            ds.SliceLocation = float(z * 3.0)
            ds.ImagePositionPatient = [0.0, 0.0, float(z * 3.0)]
            ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            ds.PixelSpacing = [0.5, 0.5]
            ds.Rows, ds.Columns = vol.shape[1], vol.shape[2]
            ds.PhotometricInterpretation = 'MONOCHROME2'
            ds.SamplesPerPixel = 1
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0
            ds.RescaleSlope = 1.0
            ds.RescaleIntercept = 0.0

            # Convert [0, 1] float to uint16
            pixel_int16 = (vol[z] * 4095.0).astype(np.uint16)
            ds.PixelData = pixel_int16.tobytes()

            ds.save_as(series_dir / f"slice_{z:03d}.dcm", write_like_original=False)

        return series_dir
