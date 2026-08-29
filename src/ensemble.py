"""
Ensembling, Pathology Blending, Probability Calibration & Submission Pipeline
==============================================================================
Provides:
1. PathologySpecificBlender: Optimal pathology-by-pathology blending of Vision + NLP probabilities.
2. ProbabilityCalibrator: Platt and temperature scaling for multi-label confidence calibration.
3. CrossValidationEnsemble: 5-Fold out-of-fold and test prediction aggregator.
4. SubmissionGenerator: Competition-compliant submission file builder with schema verification.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from scipy.optimize import minimize_scalar

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import TARGET_COLS


class PathologySpecificBlender:
    """
    Learns and applies pathology-specific optimal interpolation weights (alpha_c)
    between Vision Model predictions and NLP Clinical Extractor predictions:
        P_blend[c] = alpha_c * P_vision[c] + (1 - alpha_c) * P_nlp[c]
    
    Optimizing alpha_c independently per pathology maximizes the overall Macro-AUC,
    allowing high-performing NLP rules (e.g. MCL, Lateral Meniscus) and high-performing
    visual patterns to synergize perfectly.
    """

    def __init__(self, default_alpha: float = 0.5):
        self.default_alpha = default_alpha
        self.weights = {col: default_alpha for col in TARGET_COLS}
        self.fitted = False

    def fit(
        self, 
        y_true: np.ndarray, 
        y_prob_vision: np.ndarray, 
        y_prob_nlp: np.ndarray
    ) -> Dict[str, float]:
        """
        Optimizes alpha_c for each pathology using 1D grid search on gold validation cases.
        y_true, y_prob_vision, y_prob_nlp: each shape (N, 12)
        """
        num_classes = y_true.shape[1]
        alphas_grid = np.linspace(0.0, 1.0, 21) # Steps of 0.05

        for i in range(num_classes):
            target_name = TARGET_COLS[i] if i < len(TARGET_COLS) else f"Class_{i}"
            col_true = (y_true[:, i] >= 0.5).astype(int)
            col_vis = y_prob_vision[:, i]
            col_nlp = y_prob_nlp[:, i]

            # If class has both positive and negative examples
            if len(np.unique(col_true)) > 1:
                best_alpha = self.default_alpha
                best_auc = -1.0

                for a in alphas_grid:
                    blended = a * col_vis + (1.0 - a) * col_nlp
                    try:
                        auc = roc_auc_score(col_true, blended)
                    except Exception:
                        auc = 0.5

                    if auc > best_auc:
                        best_auc = auc
                        best_alpha = float(a)

                self.weights[target_name] = round(best_alpha, 3)
            else:
                self.weights[target_name] = self.default_alpha

        self.fitted = True
        return self.weights

    def blend(self, y_prob_vision: np.ndarray, y_prob_nlp: np.ndarray) -> np.ndarray:
        """
        Applies fitted weights to produce optimal blended predictions of shape (N, 12).
        """
        blended = np.zeros_like(y_prob_vision, dtype=np.float32)
        num_classes = y_prob_vision.shape[1]

        for i in range(num_classes):
            target_name = TARGET_COLS[i] if i < len(TARGET_COLS) else f"Class_{i}"
            alpha = self.weights.get(target_name, self.default_alpha)
            blended[:, i] = alpha * y_prob_vision[:, i] + (1.0 - alpha) * y_prob_nlp[:, i]

        return np.clip(blended, 0.0, 1.0)


class ProbabilityCalibrator:
    """
    Calibrates predicted probabilities using temperature scaling and Platt logistic regression.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = max(1e-3, temperature)

    def calibrate_temperature(self, probs: np.ndarray) -> np.ndarray:
        """
        Applies temperature scaling to probabilities in logit space.
        """
        eps = 1e-7
        clipped = np.clip(probs, eps, 1.0 - eps)
        logits = np.log(clipped / (1.0 - clipped))
        scaled_logits = logits / self.temperature
        return 1.0 / (1.0 + np.exp(-scaled_logits))

    def fit_temperature(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Optimizes single temperature scalar to minimize cross-entropy loss.
        """
        eps = 1e-7
        y_prob = np.clip(y_prob, eps, 1.0 - eps)
        logits = np.log(y_prob / (1.0 - y_prob))

        def nll_func(temp):
            scaled_logits = logits / max(1e-2, temp)
            p = 1.0 / (1.0 + np.exp(-scaled_logits))
            p = np.clip(p, eps, 1.0 - eps)
            loss = -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)).mean()
            return loss

        res = minimize_scalar(nll_func, bounds=(0.1, 5.0), method='bounded')
        self.temperature = float(res.x)
        return self.temperature


class CrossValidationEnsemble:
    """
    Aggregates multi-fold vision model predictions.
    Supports mean averaging and rank-averaged probability blending.
    """

    @staticmethod
    def average_fold_predictions(fold_predictions: List[np.ndarray]) -> np.ndarray:
        """
        Averages a list of probability matrices [(N, 12), (N, 12), ...] across folds.
        """
        if not fold_predictions:
            raise ValueError("No fold predictions provided.")
        stacked = np.stack(fold_predictions, axis=0) # (K, N, 12)
        return np.mean(stacked, axis=0)

    @staticmethod
    def rank_average_predictions(fold_predictions: List[np.ndarray]) -> np.ndarray:
        """
        Performs percentile rank averaging across folds.
        """
        from scipy.stats import rankdata
        n_folds = len(fold_predictions)
        rank_sum = np.zeros_like(fold_predictions[0], dtype=np.float32)

        for preds in fold_predictions:
            for c in range(preds.shape[1]):
                col_ranks = rankdata(preds[:, c]) / len(preds)
                rank_sum[:, c] += col_ranks

        return rank_sum / n_folds


class SubmissionGenerator:
    """
    Formats, validates, and writes the competition-ready submission CSV file.
    """

    @staticmethod
    def create_submission_dataframe(
        study_ids: List[str], 
        probabilities: np.ndarray
    ) -> pd.DataFrame:
        """
        Builds DataFrame conforming to [StudyInstanceUID, 12 Target Classes].
        """
        if len(study_ids) != probabilities.shape[0]:
            raise ValueError(f"Study IDs count ({len(study_ids)}) != Probs count ({probabilities.shape[0]})")
        if probabilities.shape[1] != len(TARGET_COLS):
            raise ValueError(f"Probs columns ({probabilities.shape[1]}) != Target classes count ({len(TARGET_COLS)})")

        sub_df = pd.DataFrame({'StudyInstanceUID': study_ids})
        for i, col in enumerate(TARGET_COLS):
            sub_df[col] = np.round(np.clip(probabilities[:, i], 0.0, 1.0), 4)

        return sub_df

    @staticmethod
    def validate_and_save(
        submission_df: pd.DataFrame, 
        output_path: Union[str, Path] = 'outputs/submission.csv'
    ) -> Path:
        """
        Runs comprehensive integrity checks and saves submission CSV.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Check columns
        expected_cols = ['StudyInstanceUID'] + TARGET_COLS
        assert list(submission_df.columns) == expected_cols, f"Columns mismatch: {submission_df.columns}"

        # 2. Check nulls / NaNs
        null_count = submission_df.isnull().sum().sum()
        assert null_count == 0, f"Found {null_count} NaN/null values in submission!"

        # 3. Check probability range [0.0, 1.0]
        numeric_vals = submission_df[TARGET_COLS].values
        assert (numeric_vals >= 0.0).all() and (numeric_vals <= 1.0).all(), "Values outside [0.0, 1.0] range!"

        # 4. Check unique studies
        assert submission_df['StudyInstanceUID'].nunique() == len(submission_df), "Duplicate StudyInstanceUIDs found!"

        submission_df.to_csv(output_path, index=False)
        print(f"[SUCCESS] Verified submission saved to: {output_path}")
        print(f"          Total rows: {len(submission_df):,} | Columns: {len(submission_df.columns)}")
        return output_path

