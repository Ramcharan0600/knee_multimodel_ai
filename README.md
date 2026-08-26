# 🦵 RSNA Knee Multimodal AI Challenge

An end-to-end multimodal machine learning system for automated detection of **12 clinically important knee abnormalities** from multi-sequence MRI studies paired with multilingual radiology reports.

---

## 🎯 Target Pathologies (12 Multi-Label Classes)

Models predict confidence scores across 12 pathologies, evaluated via **Macro-Averaged AUC-ROC**:
1. `ACL` (Anterior Cruciate Ligament tear)
2. `MCL` (Medial Collateral Ligament tear)
3. `Medial Meniscus` (tear / degeneration)
4. `Lateral Meniscus` (tear / degeneration)
5. `Medial OA` (Medial Osteoarthritis)
6. `Lateral OA` (Lateral Osteoarthritis)
7. `PF OA` (Patellofemoral Osteoarthritis)
8. `Effusion` (Joint effusion / excess fluid)
9. `Synovitis` (Synovial membrane inflammation)
10. `Baker's` (Baker's / Popliteal cyst)
11. `Contusion` (Bone marrow contusion / edema)
12. `Fracture` (Occult / acute bone fracture)

---

## 🗺️ Project Roadmap & Status

| Phase | Description | Status | Deliverables |
| :--- | :--- | :---: | :--- |
| **Phase 1** | **Exploratory Data Analysis & Validation** | ✅ Completed | • [`src/eda_analysis.py`](file:///d:/knee_multimodel_ai/src/eda_analysis.py)<br>• [`notebooks/01_eda.ipynb`](file:///d:/knee_multimodel_ai/notebooks/01_eda.ipynb)<br>• 5-Fold Stratified Split: [`data/train_folds.csv`](file:///d:/knee_multimodel_ai/data/train_folds.csv) |
| **Phase 2** | **Multilingual Report NLP & Weak Supervision** | ✅ Completed | • [`src/report_extractor.py`](file:///d:/knee_multimodel_ai/src/report_extractor.py)<br>• [`notebooks/02_report_nlp_extractor.ipynb`](file:///d:/knee_multimodel_ai/notebooks/02_report_nlp_extractor.ipynb)<br>• Pseudo-Labeled Dataset ($N=4,407$): [`data/train_pseudo_labeled.csv`](file:///d:/knee_multimodel_ai/data/train_pseudo_labeled.csv) |
| **Phase 3** | **MRI DICOM & Image Preprocessing Pipeline** | ⏳ Next | • Multi-series volume loader (`Sagittal`, `Coronal`, `Axial`)<br>• PyTorch Dataset & DataLoaders with 2.5D/3D augmentations |
| **Phase 4** | **Multi-View Vision Model Training** | ⏳ Upcoming | • Multi-sequence backbones (ConvNeXt / EfficientNet / Swin)<br>• 5-Fold Cross-Validation & Macro-AUC Optimization |
| **Phase 5** | **Multimodal Integration & Distillation** | ⏳ Upcoming | • Vision-Language contrastive alignment / feature distillation |
| **Phase 6** | **Ensembling & Submission Pipeline** | ⏳ Upcoming | • Model blending, probability calibration, submission file generator |

---

## 📊 Dataset Structure & Statistics

- **Total Studies**: 4,407 knee MRI studies
- **Gold-Standard Manual Annotations**: 58 studies (1.32%)
- **Free-Text Reports for Weak Supervision / Distillation**: 4,349 studies (98.68%)
- **Total MRI Series**: 24,371 series mappings across `Sagittal`, `Coronal`, and `Axial` planes with `Fluid_Sensitive` and `Fat_Suppression` metadata.

### Gold-Standard Target Prevalence (N=58):
- **Effusion**: 60.34% (35 cases)
- **Synovitis**: 46.55% (27 cases)
- **Medial Meniscus**: 44.83% (26 cases)
- **ACL**: 41.38% (24 cases)
- **Lateral Meniscus**: 39.66% (23 cases)
- **PF OA**: 36.21% (21 cases)
- **Contusion**: 32.76% (19 cases)
- **Fracture**: 31.03% (18 cases)
- **Medial OA**: 25.86% (15 cases)
- **Baker's**: 20.69% (12 cases)
- **Lateral OA**: 18.97% (11 cases)
- **MCL**: 15.52% (9 cases)

---

## 📝 Phase 2 NLP Extractor Benchmark Metrics (N=58 Gold Cases)

| Target Pathology | Gold Positives | Predicted Positives | Precision | Recall | F1-Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MCL** | 9 | 11 | 0.636 | 0.778 | 0.700 | **0.851** |
| **Lateral Meniscus** | 23 | 21 | 0.762 | 0.696 | 0.727 | **0.793** |
| **Lateral OA** | 11 | 12 | 0.583 | 0.636 | 0.609 | **0.781** |
| **Baker's** | 12 | 14 | 0.571 | 0.667 | 0.615 | **0.777** |
| **ACL** | 24 | 18 | 0.778 | 0.583 | 0.667 | **0.727** |
| **Fracture** | 18 | 9 | 0.778 | 0.389 | 0.519 | **0.679** |
| **Medial OA** | 15 | 12 | 0.583 | 0.467 | 0.519 | **0.668** |
| **Medial Meniscus** | 26 | 17 | 0.706 | 0.462 | 0.558 | **0.649** |
| **PF OA** | 21 | 12 | 0.667 | 0.381 | 0.485 | **0.636** |
| **Contusion** | 19 | 22 | 0.455 | 0.526 | 0.488 | **0.615** |
| **Synovitis** | 27 | 13 | 0.692 | 0.333 | 0.450 | **0.595** |
| **Effusion** | 35 | 31 | 0.645 | 0.571 | 0.606 | **0.560** |
| **Macro Average** | — | — | **0.655** | **0.541** | **0.579** | **0.694** |

---

## 📁 Repository Structure

```text
├── data/
│   ├── train.csv                 # Full cohort with reports and gold labels
│   ├── train_series.csv          # Study to series mapping & anatomical planes
│   ├── train_folds.csv           # Leak-free 5-fold multi-label stratified split
│   └── train_pseudo_labeled.csv  # 4,407 studies with calibrated NLP pseudo-labels
├── notebooks/
│   ├── 01_eda.ipynb              # Interactive Exploratory Data Analysis
│   └── 02_report_nlp_extractor.ipynb # Multilingual NLP Extractor & Benchmark
├── outputs/
│   ├── eda_plots/                # High-resolution EDA visualization plots
│   └── nlp_eval/                 # NLP benchmark metrics on Gold Standard
├── src/
│   ├── eda_analysis.py           # Automated EDA batch processor
│   ├── validation.py             # Iterative multi-label stratified fold partitioner
│   └── report_extractor.py       # Multilingual NLP extractor & weak supervision
└── requirements.txt              # Python dependencies
```

---

## 🚀 Quickstart & Reproduction

### 1. Run Automated EDA Analysis:
```bash
python src/eda_analysis.py
```

### 2. Run Multilingual NLP Extractor & Generate Pseudo-Labels:
```bash
python src/report_extractor.py
```

### 3. Regenerate Cross-Validation Folds:
```bash
python src/validation.py
```

### 4. Launch Interactive Notebooks:
Open [`notebooks/01_eda.ipynb`](file:///d:/knee_multimodel_ai/notebooks/01_eda.ipynb) or [`notebooks/02_report_nlp_extractor.ipynb`](file:///d:/knee_multimodel_ai/notebooks/02_report_nlp_extractor.ipynb) in your Jupyter/IDE environment.
