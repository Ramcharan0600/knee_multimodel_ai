"""
Predict 12 Knee Pathologies from Uploaded Image, URL, or File Explorer
======================================================================
Usage:
  1. Open Windows File Picker popup to choose any image from your computer:
     python src/predict_image.py --browse

  2. Pass a specific image file or URL directly:
     python src/predict_image.py --image "C:/Users/your_name/Downloads/knee_mri.jpg"
     python src/predict_image.py --image "https://example.com/knee_scan.png"

  3. Run without arguments to choose an option interactively:
     python src/predict_image.py
"""

import sys
import os
import argparse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import MultiViewKneeModel, TARGET_COLS
from src.mri_preprocessor import VolumeResampler, DICOMReader


def open_file_browser() -> Optional[str]:
    """
    Opens a native Windows file explorer dialog to select an image or DICOM file.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw() # Hide root window
        root.attributes('-topmost', True) # Bring file dialog to front
        
        file_path = filedialog.askopenfilename(
            title="Select Knee MRI Image or DICOM File",
            filetypes=[
                ("Medical Images & Photos", "*.png *.jpg *.jpeg *.dcm *.dicom *.bmp *.tiff"),
                ("All Files", "*.*")
            ]
        )
        root.destroy()
        return file_path if file_path else None
    except Exception as e:
        print(f"[INFO] GUI File picker not available ({e}).", flush=True)
        return None


def load_image(source: str) -> np.ndarray:
    """
    Downloads an online URL or loads a local file.
    Returns 2D or 3D numpy array normalized in [0.0, 1.0].
    """
    temp_dir = Path("outputs/temp_downloads")
    temp_dir.mkdir(parents=True, exist_ok=True)

    if source.startswith("http://") or source.startswith("https://"):
        print(f"[1/4] Downloading image from URL: {source}", flush=True)
        ext = os.path.splitext(source.split("?")[0])[1]
        if not ext or ext.lower() not in ['.jpg', '.jpeg', '.png', '.dcm']:
            ext = '.jpg'
        local_file = temp_dir / f"downloaded_mri{ext}"
        
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(source, headers=headers)
        with urllib.request.urlopen(req) as resp, open(local_file, 'wb') as out_f:
            out_f.write(resp.read())
        target_file = local_file
        print(f"      Saved locally to: {target_file}", flush=True)
    else:
        target_file = Path(source)
        if not target_file.exists():
            raise FileNotFoundError(f"File not found: {target_file}")
        print(f"[1/4] Loading local file: {target_file}", flush=True)

    ext = target_file.suffix.lower()
    if ext in ['.dcm', '.dicom']:
        pixel_array, _ = DICOMReader.read_dicom_file(target_file)
        normalized = DICOMReader.normalize_intensity(pixel_array)
    else:
        img = Image.open(target_file).convert('L')
        img_array = np.array(img, dtype=np.float32)
        v_min, v_max = img_array.min(), img_array.max()
        if v_max > v_min:
            normalized = (img_array - v_min) / (v_max - v_min)
        else:
            normalized = np.zeros_like(img_array)

    return normalized


def run_prediction(image_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Formats image into 3D multi-plane tensor and runs vision model inference.
    """
    print("[2/4] Formatting 3D Tensor [1, 3, Depth=16, H=224, W=224]...", flush=True)

    if image_array.ndim == 2:
        vol_slices = []
        for i in range(16):
            scale = 1.0 - 0.03 * abs(i - 7.5)
            resized = Image.fromarray((image_array * 255.0).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
            slice_arr = (np.array(resized, dtype=np.float32) / 255.0) * scale
            vol_slices.append(slice_arr)
        volume_3d = np.stack(vol_slices, axis=0)
    else:
        volume_3d = VolumeResampler.preprocess_volume(image_array, target_depth=16, target_spatial=(224, 224))

    tri_plane = np.stack([volume_3d, volume_3d, volume_3d], axis=0)
    input_tensor = torch.tensor(tri_plane, dtype=torch.float32).unsqueeze(0)

    print("[3/4] Running Multi-View Vision Model Forward Pass...", flush=True)
    model = MultiViewKneeModel(num_classes=12, feature_dim=128, fused_dim=256)
    model.eval()

    with torch.no_grad():
        logits, attn_maps = model(input_tensor, return_attention=True)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    attn_weights = attn_maps['sagittal'].cpu().numpy()[0]
    return probs, attn_weights, volume_3d


def save_diagnostic_visualization(volume_3d: np.ndarray, probs: np.ndarray, attn_weights: np.ndarray, output_path: str = "outputs/prediction_result.png"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    top_slice_idx = int(np.argmax(attn_weights))

    # 1. Image Slice Viewer
    axes[0].imshow(volume_3d[top_slice_idx], cmap='bone')
    axes[0].set_title(f"Diagnostic Focus Slice #{top_slice_idx+1}/16\n(Attention: {attn_weights[top_slice_idx]:.3f})", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    # 2. Slice Attention Distribution
    axes[1].bar(range(1, 17), attn_weights, color='teal', edgecolor='black')
    axes[1].axvline(top_slice_idx + 1, color='red', linestyle='--', label=f'Peak: Slice #{top_slice_idx+1}')
    axes[1].set_title("Slice Attention Distribution", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Slice Index (1 to 16)")
    axes[1].set_ylabel("Learned Attention Weight")
    axes[1].legend()

    # 3. 12 Pathology Probabilities Bar Chart
    colors = ['crimson' if p >= 0.5 else 'steelblue' for p in probs]
    y_pos = np.arange(len(TARGET_COLS))
    axes[2].barh(y_pos, probs, color=colors, edgecolor='black')
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(TARGET_COLS, fontsize=9)
    axes[2].axvline(0.5, color='black', linestyle=':', label='Positive Threshold (0.50)')
    axes[2].set_xlim(0, 1.0)
    axes[2].set_title("Predicted Pathology Confidence", fontsize=11, fontweight='bold')
    axes[2].set_xlabel("Probability [0.0 to 1.0]")
    axes[2].legend(loc='lower right')
    axes[2].invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[4/4] Saved Visual Diagnostic Report to: {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Predict 12 Knee Pathologies from Uploaded Image, URL, or File Picker")
    parser.add_argument("--image", "-i", type=str, default=None, help="Online image URL or local file path")
    parser.add_argument("--browse", "-b", action="store_true", help="Open file dialog window to pick an image from your computer")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("[IMAGE PREDICTION] RSNA KNEE MULTIMODAL AI - VISION INFERENCE TOOL", flush=True)
    print("=" * 70, flush=True)

    image_source = args.image

    if args.browse or (not image_source and sys.stdin.isatty()):
        print("\nChoose an input method:")
        print("  1. Browse and pick an image from your computer (File Explorer)")
        print("  2. Type or paste an Image File Path or Online URL")
        print("  3. Use default sample Knee MRI scan (data/sample_knee_mri.png)")
        
        try:
            choice = input("\nEnter choice (1/2/3) [default: 1]: ").strip()
        except Exception:
            choice = '3'

        if choice == '1' or choice == '':
            print("[INFO] Opening Windows File Picker dialog...", flush=True)
            picked_file = open_file_browser()
            if picked_file:
                image_source = picked_file
            else:
                print("[INFO] No file selected. Falling back to default sample scan.", flush=True)
                image_source = "data/sample_knee_mri.png"
        elif choice == '2':
            user_path = input("Enter image path or URL: ").strip()
            image_source = user_path if user_path else "data/sample_knee_mri.png"
        else:
            image_source = "data/sample_knee_mri.png"
    elif not image_source:
        image_source = "data/sample_knee_mri.png"

    if not Path(image_source).exists() and not image_source.startswith("http"):
        print(f"[ERROR] Specified file not found: {image_source}", flush=True)
        return

    image_array = load_image(image_source)
    probs, attn_weights, volume_3d = run_prediction(image_array)

    print("\n" + "=" * 70, flush=True)
    print("[RESULTS] PREDICTION RESULTS ACROSS 12 TARGET PATHOLOGIES", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Target Pathology':<22} | {'Probability':<12} | {'Diagnostic Call':<12} | Visual Bar", flush=True)
    print("-" * 70, flush=True)

    for i, target in enumerate(TARGET_COLS):
        prob = float(probs[i])
        call = "[POSITIVE]" if prob >= 0.5 else "[NEGATIVE]"
        bar_len = int(round(prob * 20))
        bar = "#" * bar_len + "-" * (20 - bar_len)
        print(f"{target:<22} | {prob:.4f}       | {call:<12} | [{bar}]", flush=True)

    print("=" * 70, flush=True)

    out_img = "outputs/prediction_result.png"
    save_diagnostic_visualization(volume_3d, probs, attn_weights, output_path=out_img)
    print(f"[DONE] Diagnostic report plot saved to: {out_img}\n", flush=True)


if __name__ == '__main__':
    main()
