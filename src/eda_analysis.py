import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid')
plt.rcParams.update({'font.size': 10, 'figure.autolayout': True})

DATA_DIR = Path('data')
OUTPUT_DIR = Path('outputs/eda_plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

def run_eda():
    print('=== RSNA Knee Multimodal AI - In-Depth Exploratory Data Analysis ===\n')
    
    # 1. Load Data
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    series_df = pd.read_csv(DATA_DIR / 'train_series.csv')
    
    labeled_df = train_df[train_df['ACL'].notna()].copy()
    unlabeled_df = train_df[train_df['ACL'].isna()].copy()
    
    print(f'Total Studies in train.csv: {len(train_df):,}')
    print(f'Studies with Gold-Standard Manual Labels: {len(labeled_df):,} ({len(labeled_df)/len(train_df)*100:.2f}%)')
    print(f'Studies with Free-Text Reports for Weak Supervision: {len(unlabeled_df):,} ({len(unlabeled_df)/len(train_df)*100:.2f}%)')
    print(f'Total MRI Series in train_series.csv: {len(series_df):,}')
    
    # 2. Gold Standard Label Analysis
    gold_counts = labeled_df[TARGET_COLS].sum().astype(int)
    gold_prev = (labeled_df[TARGET_COLS].mean() * 100).round(2)
    
    gold_stats = pd.DataFrame({
        'Positive_Count': gold_counts,
        'Prevalence_Pct': gold_prev
    }).sort_values(by='Prevalence_Pct', ascending=False)
    
    print('\n--- Gold-Standard (N=58) Target Label Distribution ---')
    print(gold_stats)
    
    # Plot 1: Gold Standard Prevalence
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=gold_stats['Prevalence_Pct'], y=gold_stats.index, palette='crest', ax=ax)
    ax.set_title(f'Gold-Standard Target Prevalence (N={len(labeled_df)} Annotated Studies)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Prevalence (%) in Annotated Cohort', fontsize=11)
    for i, v in enumerate(gold_stats['Prevalence_Pct']):
        cnt = gold_stats.loc[gold_stats.index[i], 'Positive_Count']
        ax.text(v + 0.8, i, f'{v:.1f}% ({cnt}/{len(labeled_df)})', va='center', fontsize=10, fontweight='semibold')
    ax.set_xlim(0, max(gold_stats['Prevalence_Pct']) + 12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_gold_target_prevalence.png', dpi=300)
    plt.close()
    
    # Plot 2: Correlation Heatmap in Gold Standard
    corr = labeled_df[TARGET_COLS].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', center=0, ax=ax, square=True, cbar_kws={'shrink': 0.8})
    ax.set_title('Target Co-occurrence Correlation (Gold Standard Annotations)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_gold_correlation_matrix.png', dpi=300)
    plt.close()
    
    # Plot 3: Abnormality Count per Study in Gold Standard
    labeled_df['num_pathologies'] = labeled_df[TARGET_COLS].sum(axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    path_counts = labeled_df['num_pathologies'].value_counts().sort_index()
    sns.barplot(x=path_counts.index, y=path_counts.values, palette='magma', ax=ax)
    ax.set_title('Pathology Multi-Label Count per Study (Gold Standard)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Number of Co-occurring Pathologies', fontsize=11)
    ax.set_ylabel('Number of Studies', fontsize=11)
    for i, (idx, v) in enumerate(path_counts.items()):
        ax.text(i, v + 0.5, f'{v}', ha='center', fontsize=10, fontweight='semibold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_pathology_count_distribution.png', dpi=300)
    plt.close()
    
    # Plot 4: Series Distribution per Study
    series_per_study = series_df.groupby('StudyInstanceUID').size()
    fig, ax = plt.subplots(figsize=(8, 5))
    s_counts = series_per_study.value_counts().sort_index()
    sns.barplot(x=s_counts.index, y=s_counts.values, palette='viridis', ax=ax)
    ax.set_title('MRI Series Available per Study (Total N=4,407 Studies)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Number of Series per Study', fontsize=11)
    ax.set_ylabel('Number of Studies', fontsize=11)
    for i, (idx, v) in enumerate(s_counts.items()):
        ax.text(i, v + 50, f'{v:,}', ha='center', fontsize=9, fontweight='semibold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_series_per_study.png', dpi=300)
    plt.close()
    
    # Plot 5: Anatomical Plane & Contrast Matrix
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plane_counts = series_df['Anatomical_Plane'].value_counts()
    sns.barplot(x=plane_counts.index, y=plane_counts.values, palette='Set2', ax=axes[0])
    axes[0].set_title('Series Count by Anatomical Plane', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Series Count', fontsize=11)
    for i, (idx, v) in enumerate(plane_counts.items()):
        axes[0].text(i, v + 150, f'{v:,}', ha='center', fontweight='semibold')
        
    contrast_df = series_df.groupby(['Fluid_Sensitive', 'Fat_Suppression']).size().reset_index(name='count')
    contrast_df['Label'] = [f'Fluid: {r.Fluid_Sensitive}\nFatSupp: {r.Fat_Suppression}' for r in contrast_df.itertuples()]
    sns.barplot(x='Label', y='count', data=contrast_df, palette='flare', ax=axes[1])
    axes[1].set_title('Contrast & Fat Suppression Combinations', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Series Count', fontsize=11)
    for i, (idx, v) in enumerate(contrast_df['count'].items()):
        axes[1].text(i, v + 150, f'{v:,}', ha='center', fontweight='semibold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_series_characteristics.png', dpi=300)
    plt.close()
    
    # Plot 6: Report Text Word Count Distribution
    train_df['report_words'] = train_df['Report'].fillna('').apply(lambda x: len(x.split()))
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(train_df['report_words'], bins=40, kde=True, color='teal', ax=ax)
    ax.set_title('Radiology Report Word Count Distribution (All 4,407 Studies)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Word Count per Report', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '06_report_word_count.png', dpi=300)
    plt.close()
    
    print('\nEDA processing complete. All 6 figures updated in outputs/eda_plots/')

if __name__ == '__main__':
    run_eda()
