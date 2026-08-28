"""
Model Training, Validation & Evaluation Engine for RSNA Knee Multimodal AI
==========================================================================
Provides:
1. calculate_metrics: Comprehensive multi-label metric calculation (Macro-AUC, F1, Precision, Recall).
2. ModelTrainer: Training engine with CosineAnnealing scheduling, gradient clipping, and checkpointing.
3. evaluate_model: Inference and evaluation report generator.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, log_loss
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure project root is accessible
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import TARGET_COLS


def calculate_metrics(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    threshold: float = 0.5
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Computes Macro-Averaged AUC-ROC and per-pathology performance metrics.
    y_true: (N, 12)
    y_prob: (N, 12)
    """
    num_classes = y_true.shape[1]
    per_class_auc = {}
    per_class_f1 = {}
    per_class_prec = {}
    per_class_rec = {}

    aucs = []
    f1s = []

    for i in range(num_classes):
        target_name = TARGET_COLS[i] if i < len(TARGET_COLS) else f"Class_{i}"
        col_true = y_true[:, i]
        col_prob = y_prob[:, i]

        col_bin_true = (col_true >= threshold).astype(int)
        col_bin_pred = (col_prob >= threshold).astype(int)

        if len(np.unique(col_bin_true)) > 1:
            try:
                auc = float(roc_auc_score(col_bin_true, col_prob))
            except Exception:
                auc = 0.5
        else:
            auc = 0.5

        f1 = float(f1_score(col_bin_true, col_bin_pred, zero_division=0))
        prec = float(precision_score(col_bin_true, col_bin_pred, zero_division=0))
        rec = float(recall_score(col_bin_true, col_bin_pred, zero_division=0))

        per_class_auc[target_name] = round(auc, 4)
        per_class_f1[target_name] = round(f1, 4)
        per_class_prec[target_name] = round(prec, 4)
        per_class_rec[target_name] = round(rec, 4)

        aucs.append(auc)
        f1s.append(f1)

    macro_auc = float(np.mean(aucs))
    macro_f1 = float(np.mean(f1s))

    return {
        'macro_auc': round(macro_auc, 4),
        'macro_f1': round(macro_f1, 4),
        'per_class_auc': per_class_auc,
        'per_class_f1': per_class_f1,
        'per_class_precision': per_class_prec,
        'per_class_recall': per_class_rec
    }


class ModelTrainer:
    """
    Production-grade training orchestrator for multi-view knee MRI models.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        max_grad_norm: float = 1.0,
        save_dir: Union[str, Path] = 'models',
        fold_id: int = 0
    ):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(self.device)
        
        self.optimizer = optimizer if optimizer is not None else torch.optim.AdamW(
            self.model.parameters(), lr=1e-3, weight_decay=1e-2
        )
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.fold_id = fold_id

        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_macro_auc': [],
            'val_macro_f1': [],
            'lr': []
        }
        self.best_macro_auc = -1.0
        self.best_checkpoint_path = self.save_dir / f"best_model_fold{self.fold_id}.pt"

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        for batch in self.train_loader:
            self.optimizer.zero_grad()

            image = batch['image'].to(self.device)
            targets = batch['targets'].to(self.device)
            has_gold = batch['has_gold'].to(self.device)

            logits = self.model(image)
            loss = self.criterion(logits, targets, has_gold)

            loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, Dict[str, Union[float, Dict[str, float]]]]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for batch in self.val_loader:
            image = batch['image'].to(self.device)
            targets = batch['targets'].to(self.device)
            has_gold = batch['has_gold'].to(self.device)

            logits = self.model(image)
            loss = self.criterion(logits, targets, has_gold)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(targets.cpu().numpy())

        avg_loss = total_loss / max(1, len(self.val_loader))
        y_prob = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_targets, axis=0)

        metrics = calculate_metrics(y_true, y_prob)
        return avg_loss, metrics

    def fit(self, epochs: int = 10, early_stopping_patience: int = 5) -> Dict[str, List]:
        print(f"\n=======================================================")
        print(f"[TRAIN] Training MultiViewKneeModel on Fold {self.fold_id} ({epochs} Epochs)")
        print(f"        Device: {self.device} | Batches/Epoch: Train={len(self.train_loader)}, Val={len(self.val_loader)}")
        print(f"=======================================================\n")

        patience_counter = 0

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            train_loss = self.train_epoch()
            val_loss, metrics = self.evaluate()

            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            elapsed = time.time() - start_time
            val_auc = metrics['macro_auc']
            val_f1 = metrics['macro_f1']

            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_macro_auc'].append(val_auc)
            self.history['val_macro_f1'].append(val_f1)
            self.history['lr'].append(current_lr)

            saved_indicator = ""
            if val_auc > self.best_macro_auc:
                self.best_macro_auc = val_auc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'macro_auc': val_auc,
                    'metrics': metrics
                }, self.best_checkpoint_path)
                saved_indicator = " [BEST MODEL SAVED]"
                patience_counter = 0
            else:
                patience_counter += 1

            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Macro-AUC: {val_auc:.4f} | Val F1: {val_f1:.4f} | LR: {current_lr:.1e}{saved_indicator}"
            )

            if patience_counter >= early_stopping_patience:
                print(f"\n[INFO] Early stopping triggered after {patience_counter} epochs without improvement.")
                break

        print(f"\n[Finished] Best Validation Macro-AUC: {self.best_macro_auc:.4f}")
        print(f"Checkpoint saved to: {self.best_checkpoint_path}\n")
        return self.history


def evaluate_model(
    model: nn.Module, 
    dataloader: DataLoader, 
    device: Optional[torch.device] = None
) -> Tuple[pd.DataFrame, float]:
    """
    Evaluates a trained model and returns a formatted DataFrame report of metrics per pathology.
    """
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            image = batch['image'].to(device)
            targets = batch['targets']

            logits = model(image)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_preds.append(probs)
            all_targets.append(targets.numpy())

    y_prob = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    metrics = calculate_metrics(y_true, y_prob)

    report_rows = []
    for path in TARGET_COLS:
        report_rows.append({
            'Pathology': path,
            'AUC-ROC': metrics['per_class_auc'].get(path, 0.0),
            'F1-Score': metrics['per_class_f1'].get(path, 0.0),
            'Precision': metrics['per_class_precision'].get(path, 0.0),
            'Recall': metrics['per_class_recall'].get(path, 0.0)
        })

    report_df = pd.DataFrame(report_rows)
    return report_df, metrics['macro_auc']
