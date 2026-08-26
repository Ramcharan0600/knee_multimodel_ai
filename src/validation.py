import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path('data')
TARGET_COLS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

def iterative_multilabel_stratification(df, target_cols, n_splits=5, seed=42):
    """
    Performs greedy iterative multi-label stratification.
    Guarantees balanced distribution of multiple binary labels across folds.
    """
    np.random.seed(seed)
    y = df[target_cols].values
    n_samples, n_labels = y.shape
    
    # Track assigned fold indices
    folds = np.full(n_samples, -1, dtype=int)
    
    # Desired positive count per fold for each label
    desired_pos_per_fold = np.tile(np.sum(y, axis=0) / n_splits, (n_splits, 1))
    current_pos_per_fold = np.zeros((n_splits, n_labels))
    samples_per_fold = np.zeros(n_splits)
    target_samples_per_fold = n_samples / n_splits
    
    # Sort samples by label rarity (prioritize rare labels first)
    label_rarity = np.sum(y, axis=0)
    sample_scores = np.dot(y, 1.0 / (label_rarity + 1e-5))
    order = np.argsort(-sample_scores)
    
    for idx in order:
        sample_y = y[idx]
        if np.sum(sample_y) == 0:
            # If negative for all labels, assign to fold with fewest samples
            best_fold = np.argmin(samples_per_fold)
        else:
            # Calculate score for each fold based on needed positive labels
            scores = []
            for f in range(n_splits):
                needed = desired_pos_per_fold[f] - current_pos_per_fold[f]
                score = np.sum(sample_y * needed) - 0.01 * (samples_per_fold[f] / target_samples_per_fold)
                scores.append(score)
            best_fold = np.argmax(scores)
            
        folds[idx] = best_fold
        current_pos_per_fold[best_fold] += sample_y
        samples_per_fold[best_fold] += 1
        
    df_out = df.copy()
    df_out['fold'] = folds
    return df_out

def create_folds():
    print('=== Generating Leak-Free Multi-Label Stratified Folds ===\n')
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    
    labeled_df = train_df[train_df['ACL'].notna()].copy().reset_index(drop=True)
    unlabeled_df = train_df[train_df['ACL'].isna()].copy().reset_index(drop=True)
    
    # 1. Stratify Labeled Set
    stratified_labeled = iterative_multilabel_stratification(labeled_df, TARGET_COLS, n_splits=5, seed=42)
    
    # 2. Randomly partition unlabelled studies across the same 5 folds
    np.random.seed(42)
    unlabeled_df['fold'] = np.random.randint(0, 5, size=len(unlabeled_df))
    
    # Combine back into unified DataFrame
    combined_folds = pd.concat([stratified_labeled, unlabeled_df], ignore_index=True)
    
    # Verification of label balance across folds on the gold standard set
    print('--- Gold Standard Class Count by Fold ---')
    gold_by_fold = stratified_labeled.groupby('fold')[TARGET_COLS].sum().astype(int)
    print(gold_by_fold)
    
    print('\n--- Total Studies per Fold ---')
    print(combined_folds['fold'].value_counts().sort_index())
    
    # Save output
    out_path = DATA_DIR / 'train_folds.csv'
    combined_folds.to_csv(out_path, index=False)
    print(f'\nSaved stratified folds to: {out_path}')

if __name__ == '__main__':
    create_folds()
