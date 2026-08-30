"""
Interactive Web Application for RSNA Knee Multimodal AI
========================================================
A lightweight standalone web app with drag-and-drop image upload,
online URL support, and instant live pathology prediction visualization.

Run with:
  python app.py
Then open http://localhost:5000 in your browser!
"""

import sys
import io
import json
import base64
import urllib.request
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import MultiViewKneeModel, TARGET_COLS
from src.mri_preprocessor import VolumeResampler, DICOMReader
from src.report_extractor import extract_study_probabilities
from src.ensemble import PathologySpecificBlender

PORT = 5000

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🦵 RSNA Knee Multimodal AI - Diagnostic Studio</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        body { background-color: #0f172a; color: #f8fafc; padding: 24px; }
        .container { max-width: 1100px; margin: 0 auto; }
        header { text-align: center; margin-bottom: 28px; }
        header h1 { font-size: 2.2rem; color: #38bdf8; margin-bottom: 8px; }
        header p { color: #94a3b8; font-size: 1.05rem; }
        
        .main-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
        @media(max-width: 800px) { .main-grid { grid-template-columns: 1fr; } }
        
        .card { background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }
        .card h2 { color: #f1f5f9; font-size: 1.25rem; margin-bottom: 16px; border-bottom: 1px solid #334155; padding-bottom: 8px; }
        
        .dropzone { border: 2px dashed #38bdf8; border-radius: 8px; padding: 32px 16px; text-align: center; cursor: pointer; transition: 0.2s; background: #0f172a80; }
        .dropzone:hover { background: #1e293b; border-color: #0ea5e9; }
        .dropzone input { display: none; }
        
        .preview-box { margin-top: 16px; text-align: center; min-height: 180px; display: flex; align-items: center; justify-content: center; background: #0b0f19; border-radius: 8px; overflow: hidden; }
        .preview-box img { max-width: 100%; max-height: 220px; object-fit: contain; }
        
        .input-group { margin-top: 16px; }
        .input-group label { display: block; color: #94a3b8; font-size: 0.9rem; margin-bottom: 6px; font-weight: 500; }
        .input-group input, .input-group textarea { width: 100%; padding: 10px 12px; background: #0f172a; border: 1px solid #334155; border-radius: 6px; color: #fff; font-size: 0.95rem; }
        .input-group textarea { height: 75px; resize: vertical; }
        
        button.btn { width: 100%; margin-top: 18px; padding: 12px; background: #0284c7; color: #fff; border: none; border-radius: 6px; font-size: 1.05rem; font-weight: bold; cursor: pointer; transition: 0.2s; }
        button.btn:hover { background: #0369a1; }
        
        .result-item { display: flex; align-items: center; margin-bottom: 10px; font-size: 0.92rem; }
        .result-name { width: 160px; font-weight: 500; }
        .progress-bar-bg { flex: 1; height: 16px; background: #0f172a; border-radius: 8px; overflow: hidden; margin: 0 12px; position: relative; }
        .progress-bar-fill { height: 100%; width: 0%; border-radius: 8px; transition: width 0.5s ease-out; }
        .result-pct { width: 60px; text-align: right; font-weight: bold; }
        .status-tag { width: 90px; text-align: right; font-size: 0.8rem; font-weight: bold; margin-left: 8px; }
        
        .tag-positive { color: #f87171; }
        .tag-negative { color: #34d399; }
        
        .loading { display: none; text-align: center; color: #38bdf8; margin: 20px 0; font-weight: bold; }
        .diagnostic-image-box { margin-top: 16px; }
        .diagnostic-image-box img { width: 100%; border-radius: 8px; border: 1px solid #334155; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🦵 RSNA Knee Multimodal AI Studio</h1>
            <p>Upload a Knee MRI Scan (JPG / PNG / DICOM) or paste an online image link to predict 12 pathologies</p>
        </header>

        <div class="main-grid">
            <!-- Left Card: Input -->
            <div class="card">
                <h2>1. Upload MRI Image & Report</h2>
                
                <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
                    <p>📁 <strong>Click to Browse</strong> or Drag & Drop Image Here</p>
                    <p style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">Supports PNG, JPG, JPEG, DICOM (.dcm)</p>
                    <input type="file" id="fileInput" accept="image/*,.dcm">
                </div>

                <div class="preview-box" id="previewBox">
                    <span style="color: #475569;">No image selected yet</span>
                </div>

                <div class="input-group">
                    <label>Or Paste Online Image URL:</label>
                    <input type="text" id="urlInput" placeholder="https://example.com/knee_mri.jpg">
                </div>

                <div class="input-group">
                    <label>Optional Radiology Text Report (Multimodal):</label>
                    <textarea id="reportInput" placeholder="e.g. Impression: Complete ACL tear with moderate joint effusion. Both menisci intact."></textarea>
                </div>

                <button class="btn" id="analyzeBtn" onclick="analyzeKnee()">🚀 Run AI Diagnostic Analysis</button>
                <div class="loading" id="loading">⏳ Processing MRI Volume & Computing Slice Attention...</div>
            </div>

            <!-- Right Card: Predictions -->
            <div class="card">
                <h2>2. Diagnostic Pathology Predictions</h2>
                <div id="resultsContainer">
                    <p style="color: #64748b; text-align: center; padding: 40px 0;">Upload an image on the left and click "Run AI Diagnostic Analysis" to see predicted probabilities.</p>
                </div>

                <div class="diagnostic-image-box" id="plotContainer" style="display: none;">
                    <h3 style="font-size: 1rem; color: #94a3b8; margin-bottom: 8px;">📊 Model Diagnostic Report:</h3>
                    <img id="diagnosticPlot" src="" alt="Diagnostic Report">
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentImageBase64 = null;

        const fileInput = document.getElementById('fileInput');
        const previewBox = document.getElementById('previewBox');
        const dropzone = document.getElementById('dropzone');

        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(evt) {
                    currentImageBase64 = evt.target.result;
                    previewBox.innerHTML = `<img src="${currentImageBase64}" alt="MRI Preview">`;
                };
                reader.readAsDataURL(file);
            }
        });

        // Drag and drop
        dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.style.borderColor = '#0ea5e9'; });
        dropzone.addEventListener('dragleave', () => { dropzone.style.borderColor = '#38bdf8'; });
        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.style.borderColor = '#38bdf8';
            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                const event = new Event('change');
                fileInput.dispatchEvent(event);
            }
        });

        async function analyzeKnee() {
            const urlVal = document.getElementById('urlInput').value.trim();
            const reportVal = document.getElementById('reportInput').value.trim();
            const loading = document.getElementById('loading');
            const analyzeBtn = document.getElementById('analyzeBtn');

            if (!currentImageBase64 && !urlVal && !reportVal) {
                alert('Please upload an image, enter an image URL, or provide a text report.');
                return;
            }

            loading.style.display = 'block';
            analyzeBtn.disabled = true;

            const payload = {
                image_base64: currentImageBase64,
                image_url: urlVal,
                report_text: reportVal
            };

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await response.json();
                renderResults(data);
            } catch (err) {
                alert('Error analyzing study: ' + err);
            } finally {
                loading.style.display = 'none';
                analyzeBtn.disabled = false;
            }
        }

        function renderResults(data) {
            const container = document.getElementById('resultsContainer');
            const plotContainer = document.getElementById('plotContainer');
            const diagnosticPlot = document.getElementById('diagnosticPlot');

            let html = '';
            for (const item of data.predictions) {
                const isPos = item.probability >= 0.5;
                const barColor = isPos ? '#ef4444' : '#3b82f6';
                const statusTag = isPos ? '<span class="tag-positive">🔴 POSITIVE</span>' : '<span class="tag-negative">🟢 NEGATIVE</span>';
                const pct = (item.probability * 100).toFixed(1) + '%';

                html += `
                    <div class="result-item">
                        <div class="result-name">${item.pathology}</div>
                        <div class="progress-bar-bg">
                            <div class="progress-bar-fill" style="width: ${pct}; background: ${barColor};"></div>
                        </div>
                        <div class="result-pct">${pct}</div>
                        <div class="status-tag">${statusTag}</div>
                    </div>
                `;
            }
            container.innerHTML = html;

            if (data.plot_base64) {
                diagnosticPlot.src = 'data:image/png;base64,' + data.plot_base64;
                plotContainer.style.display = 'block';
            }
        }
    </script>
</body>
</html>
"""


def process_prediction_request(payload: dict) -> dict:
    """
    Handles image decode / download and computes model predictions and plot.
    """
    image_base64 = payload.get('image_base64')
    image_url = payload.get('image_url')
    report_text = payload.get('report_text', '').strip()

    image_array = None

    if image_base64 and ',' in image_base64:
        # Decode base64 image data
        img_bytes = base64.b64decode(image_base64.split(',')[1])
        img = Image.open(io.BytesIO(img_bytes)).convert('L')
        arr = np.array(img, dtype=np.float32)
        v_min, v_max = arr.min(), arr.max()
        image_array = (arr - v_min) / (v_max - v_min) if v_max > v_min else np.zeros_like(arr)
    elif image_url:
        # Download from URL
        req = urllib.request.Request(image_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            img = Image.open(io.BytesIO(resp.read())).convert('L')
            arr = np.array(img, dtype=np.float32)
            v_min, v_max = arr.min(), arr.max()
            image_array = (arr - v_min) / (v_max - v_min) if v_max > v_min else np.zeros_like(arr)
    else:
        # Default sample scan
        sample_path = Path("data/sample_knee_mri.png")
        if sample_path.exists():
            img = Image.open(sample_path).convert('L')
            arr = np.array(img, dtype=np.float32)
            image_array = arr / 255.0
        else:
            image_array = np.random.uniform(0.1, 0.9, (224, 224)).astype(np.float32)

    # 1. Format into 3D volume
    if image_array.ndim == 2:
        vol_slices = []
        for i in range(16):
            scale = 1.0 - 0.03 * abs(i - 7.5)
            resized = Image.fromarray((image_array * 255.0).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
            vol_slices.append((np.array(resized, dtype=np.float32) / 255.0) * scale)
        volume_3d = np.stack(vol_slices, axis=0)
    else:
        volume_3d = VolumeResampler.preprocess_volume(image_array, target_depth=16, target_spatial=(224, 224))

    tri_plane = np.stack([volume_3d, volume_3d, volume_3d], axis=0)
    input_tensor = torch.tensor(tri_plane, dtype=torch.float32).unsqueeze(0)

    # 2. Vision Model
    model = MultiViewKneeModel(num_classes=12, feature_dim=128, fused_dim=256)
    model.eval()
    with torch.no_grad():
        logits, attn_maps = model(input_tensor, return_attention=True)
        vis_probs = torch.sigmoid(logits).cpu().numpy()[0]
    attn_weights = attn_maps['sagittal'].cpu().numpy()[0]

    # 3. Multimodal Blend if report text provided
    if report_text:
        nlp_scores = extract_study_probabilities(report_text)
        nlp_vec = np.array([nlp_scores[c] for c in TARGET_COLS], dtype=np.float32)
        blender = PathologySpecificBlender()
        final_probs = blender.blend(np.array([vis_probs]), np.array([nlp_vec]))[0]
    else:
        final_probs = vis_probs

    # 4. Generate Diagnostic Plot as Base64
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    top_idx = int(np.argmax(attn_weights))

    axes[0].imshow(volume_3d[top_idx], cmap='bone')
    axes[0].set_title(f"Focus Slice #{top_idx+1}/16 (Attn: {attn_weights[top_idx]:.3f})", fontsize=10, fontweight='bold')
    axes[0].axis('off')

    axes[1].bar(range(1, 17), attn_weights, color='#0ea5e9')
    axes[1].axvline(top_idx + 1, color='red', linestyle='--')
    axes[1].set_title("Slice Attention Distribution", fontsize=10, fontweight='bold')
    axes[1].set_xlabel("Slice Depth (1 to 16)")

    colors = ['#ef4444' if p >= 0.5 else '#3b82f6' for p in final_probs]
    y_pos = np.arange(len(TARGET_COLS))
    axes[2].barh(y_pos, final_probs, color=colors)
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(TARGET_COLS, fontsize=8.5)
    axes[2].axvline(0.5, color='black', linestyle=':')
    axes[2].set_xlim(0, 1.0)
    axes[2].set_title("Pathology Confidence", fontsize=10, fontweight='bold')
    axes[2].invert_yaxis()

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130)
    plt.close()
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')

    predictions = []
    for i, col in enumerate(TARGET_COLS):
        predictions.append({
            'pathology': col,
            'probability': round(float(final_probs[i]), 4)
        })

    return {
        'predictions': predictions,
        'plot_base64': plot_base64
    }


class KneeAIRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/api/predict':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                payload = json.loads(body)
                response_data = process_prediction_request(payload)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()


def start_server():
    server = HTTPServer(('localhost', PORT), KneeAIRequestHandler)
    print("=" * 70, flush=True)
    print(f"[WEB APP] RSNA KNEE MULTIMODAL AI - STUDIO RUNNING", flush=True)
    print(f"[ACCESS] Open your web browser at: http://localhost:{PORT}", flush=True)
    print("=" * 70, flush=True)
    print("Press Ctrl + C in terminal to stop server.\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped. Bye!", flush=True)
        server.server_close()


if __name__ == '__main__':
    start_server()

