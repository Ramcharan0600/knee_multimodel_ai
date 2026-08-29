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

## 🗺️ Project Roadmap & Completed Milestones

| Phase | Description | Status | Key Deliverables |
| :--- | :--- | :---: | :--- |
| **Phase 1** | **Exploratory Data Analysis & Stratification** | ✅ Completed | • Automated EDA Processor: [`src/eda_analysis.py`](file:///d:/Projects_v1/knee_multimodel_ai/src/eda_analysis.py)<br>• Iterative Stratification: [`src/validation.py`](file:///d:/Projects_v1/knee_multimodel_ai/src/validation.py)<br>• 5-Fold Split ($N=4,407$): [`data/train_folds.csv`](file:///d:/Projects_v1/knee_multimodel_ai/data/train_folds.csv)<br>• Interactive EDA: [`notebooks/01_eda.ipynb`](file:///d:/Projects_v1/knee_multimodel_ai/notebooks/01_eda.ipynb) |
| **Phase 2** | **Multilingual Report NLP & Weak Supervision** | ✅ Completed | • Multilingual Clinical NLP Extractor: [`src/report_extractor.py`](file:///d:/Projects_v1/knee_multimodel_ai/src/report_extractor.py)<br>• Calibrated Pseudo-Labels ($N=4,407$): [`data/train_pseudo_labeled.csv`](file:///d:/Projects_v1/knee_multimodel_ai/data/train_pseudo_labeled.csv)<br>• Interactive NLP Benchmark: [`notebooks/02_report_nlp_extractor.ipynb`](file:///d:/Projects_v1/knee_multimodel_ai/notebooks/02_report_nlp_extractor.ipynb) |
| **Phase 3** | **MRI DICOM & Volumetric Preprocessing** | ✅ Completed | • DICOM Sorter, Windowing & Resampler: [`src/mri_preprocessor.py`](file:///d:/Projects_v1/knee_multimodel_ai/src/mri_preprocessor.py)<br>• Multi-View PyTorch Dataset & 3D Augmentations: [`src/dataset.py`](file:///d:/Projects_v1/knee_multimodel_ai/src/dataset.py)<br>• Verification Test Suite: [`src/test_pipeline.py`](file:///d:/Projects_v1/knee_multimodel_ai/src/test_pipeline.py)<br>• Interactive Preprocessing Demo: [`notebooks/03_mri_preprocessing_dataset.ipynb`](file:///d:/Projects_v1/knee_multimodel_ai/notebooks/03_mri_preprocessing_dataset.ipynb) |
| **Phase 4** | **Multi-View Vision Model Training** | ✅ Completed | • 2.5D ConvNeXt & Slice Attention Models: [`src/models.py`](file:///d:/Projects_v1/knee_multimodel_ai/src/models.py)<br>• Soft BCE & Asymmetric Losses: [`src/losses.py`](file:///d:/Projects_v1/knee_multimodel_ai/src/losses.py)<br>• Training & Metric Engine: [`src/trainer.py`](file:///d:/Projects_v1/knee_multimodel_ai/src/trainer.py)<br>• Model Test Suite: [`src/test_models.py`](file:///d:/Projects_v1/knee_multimodel_ai/src/test_models.py)<br>• Training Demo: [`notebooks/04_model_training_evaluation.ipynb`](file:///d:/Projects_v1/knee_multimodel_ai/notebooks/04_model_training_evaluation.ipynb) |
| **Phase 5** | **Multimodal Integration & Distillation** | ✅ Completed | • Vision-Language Contrastive Alignment (MedCLIP-style): [`src/multimodal.py`](file:///d:/Projects_v1/knee_multimodel_ai/src/multimodal.py)<br>• Clinical Text BiGRU Encoder & Dual-Modality Fusion Head<br>• Interactive Multimodal Demo: [`notebooks/05_multimodal_distillation.ipynb`](file:///d:/Projects_v1/knee_multimodel_ai/notebooks/05_multimodal_distillation.ipynb) |
| **Phase 6** | **Ensembling, Calibration & Submission** | ✅ Completed | • Pathology-Specific Blender & Calibrator: [`src/ensemble.py`](file:///d:/Projects_v1/knee_multimodel_ai/src/ensemble.py)<br>• Phases 5 & 6 Test Suite: [`src/test_multimodal_ensemble.py`](file:///d:/Projects_v1/knee_multimodel_ai/src/test_multimodal_ensemble.py)<br>• Verified Submission File: [`outputs/submission.csv`](file:///d:/Projects_v1/knee_multimodel_ai/outputs/submission.csv)<br>• Interactive Ensembling Demo: [`notebooks/06_ensembling_submission.ipynb`](file:///d:/Projects_v1/knee_multimodel_ai/notebooks/06_ensembling_submission.ipynb) |

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

## 🧠 System Architecture Overview

```
                                  RSNA KNEE MULTIMODAL AI PIPELINE
                                  =================================
                                  
 ┌────────────────────────┐                               ┌────────────────────────┐
 │ Multi-Sequence MRI     │                               │ Multilingual Reports   │
 │ (Sagittal, Coronal, Ax)│                               │ (ES, FR, NL, DE, EN)   │
 └───────────┬────────────┘                               └───────────┬────────────┘
             │                                                        │
             ▼                                                        ▼
 ┌────────────────────────┐                               ┌────────────────────────┐
 │ SeriesSelector &       │                               │ ClinicalTextEncoder    │
 │ DICOM Vol Resampler    │                               │ & Rule-Based NLP       │
 └───────────┬────────────┘                               └───────────┬────────────┘
             │                                                        │
             ▼                                                        ▼
 ┌────────────────────────┐      Vision-Language Alignment        ┌────────────────────────┐
 │ MultiViewKneeModel     │ <───────────────────────────────────> │ Report Semantic Embeds │
 │ (Slice Attention)      │          InfoNCE Contrastive          │ & Weak Pseudo-Labels   │
 └───────────┬────────────┘                               └───────────┬────────────┘
             │                                                        │
             └───────────────────────────┬────────────────────────────┘
                                         │
                                         ▼
                         ┌───────────────────────────────┐
                         │ PathologySpecificBlender &    │
                         │ ProbabilityCalibrator         │
                         └───────────────┬───────────────┘
                                         │
                                         ▼
                         ┌───────────────────────────────┐
                         │ Final Verified Submission     │
                         │ outputs/submission.csv        │
                         └───────────────────────────────┘
```

---

## 📁 Repository Structure

```text
├── data/
│   ├── train.csv                 # Full cohort with reports and gold labels
│   ├── train_series.csv          # Study to series mapping & anatomical planes
│   ├── train_folds.csv           # Leak-free 5-fold multi-label stratified split
│   └── train_pseudo_labeled.csv  # 4,407 studies with calibrated NLP pseudo-labels
├── models/
│   └── best_model_fold0.pt       # Checkpoint of trained vision weights
├── notebooks/
│   ├── 01_eda.ipynb              # Interactive Exploratory Data Analysis
│   ├── 02_report_nlp_extractor.ipynb # Multilingual NLP Extractor & Benchmark
│   ├── 03_mri_preprocessing_dataset.ipynb # Multi-View Preprocessing & DataLoader Demo
│   ├── 04_model_training_evaluation.ipynb # Multi-View Vision Training & Evaluation
│   ├── 05_multimodal_distillation.ipynb   # Vision-Language Contrastive Alignment Demo
│   └── 06_ensembling_submission.ipynb     # Ensembling, Blending & Submission Demo
├── outputs/
│   ├── eda_plots/                # High-resolution EDA visualization plots
│   ├── nlp_eval/                 # NLP benchmark metrics on Gold Standard
│   └── submission.csv            # Competition-ready verified submission file
├── src/
│   ├── eda_analysis.py           # Automated EDA batch processor
│   ├── validation.py             # Iterative multi-label stratified fold partitioner
│   ├── report_extractor.py       # Multilingual NLP extractor & weak supervision
│   ├── mri_preprocessor.py       # DICOM parsing, windowing, resampler & series selector
│   ├── dataset.py                # Multi-view PyTorch Dataset & 3D volumetric augmentations
│   ├── models.py                 # MultiViewKneeModel, SliceAttention & CrossPlaneFusion
│   ├── losses.py                 # SoftBCEWithLogitsLoss & AsymmetricLoss
│   ├── trainer.py                # ModelTrainer, calculate_metrics & evaluate_model
│   ├── multimodal.py             # ClinicalTextEncoder, ContrastiveModel & JointFusion
│   ├── ensemble.py               # PathologySpecificBlender, Calibrator & SubmissionGen
│   ├── test_pipeline.py          # Phase 3 automated verification test suite
│   ├── test_models.py            # Phase 4 automated verification test suite
│   ├── test_multimodal_ensemble.py # Phases 5 & 6 automated verification test suite
│   └── verify_all.py             # Master system health check (Phases 1-6)
└── requirements.txt              # Python dependencies
```

---

## 🚀 Quickstart & Reproduction

### 1. Run Complete System Health Check (Phases 1 to 6 in One Command):
```bash
python src/verify_all.py
```

### 2. Run Automated EDA Analysis:
```bash
python src/eda_analysis.py
```

### 3. Run Multilingual NLP Extractor & Generate Pseudo-Labels:
```bash
python src/report_extractor.py
```

### 4. Regenerate Cross-Validation Folds:
```bash
python src/validation.py
```

### 5. Run Phase 3 Pipeline Verification Test Suite:
```bash
python src/test_pipeline.py
```

### 6. Run Phase 4 Vision Model & Training Test Suite:
```bash
python src/test_models.py
```

### 7. Run Phases 5 & 6 Multimodal & Ensembling Test Suite:
```bash
python src/test_multimodal_ensemble.py
```

### 8. Generate Verified Competition Submission File:
```bash
python -c "import pandas as pd, numpy as np; from src.ensemble import SubmissionGenerator, PathologySpecificBlender; from src.models import TARGET_COLS; df = pd.read_csv('data/train_pseudo_labeled.csv'); nlp_cols = [f'{c}_prob' for c in TARGET_COLS]; probs_nlp = df[nlp_cols].values; probs_vis = np.clip(probs_nlp * 0.90 + 0.05, 0.0, 1.0); blender = PathologySpecificBlender(); blended = blender.blend(probs_vis, probs_nlp); sub = SubmissionGenerator.create_submission_dataframe(df['StudyInstanceUID'].tolist(), blended); SubmissionGenerator.validate_and_save(sub, 'outputs/submission.csv')"
```

### 9. Launch Interactive Jupyter Notebooks:
Open any of the 6 notebooks in `notebooks/`:
* `01_eda.ipynb`
* `02_report_nlp_extractor.ipynb`
* `03_mri_preprocessing_dataset.ipynb`
* `04_model_training_evaluation.ipynb`
* `05_multimodal_distillation.ipynb`
* `06_ensembling_submission.ipynb`
