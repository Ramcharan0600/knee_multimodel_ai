"""
Interactive Manual Testing CLI for RSNA Knee Multimodal AI
==========================================================
Allows you to manually test:
1. Custom Radiology Report Text: Paste any sentence in English, Spanish, French, Dutch, German.
2. Dataset Study Explorer: Inspect predictions on any of the 4,407 studies in the dataset.
3. Custom Pathology Checklist: Test how specific combinations of abnormalities are processed.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import MultiViewKneeModel, TARGET_COLS
from src.report_extractor import extract_study_probabilities, split_clauses, is_negated
from src.ensemble import PathologySpecificBlender


def print_score_table(scores_dict: dict, title: str = "PREDICTION BREAKDOWN"):
    print("\n" + "=" * 65)
    print(f"[*] {title}")
    print("=" * 65)
    print(f"{'Target Pathology':<22} | {'Probability':<12} | {'Status':<12} | Visual Bar")
    print("-" * 65)

    for col in TARGET_COLS:
        prob = scores_dict.get(col, 0.0)
        status = "[POSITIVE]" if prob >= 0.5 else "[NEGATIVE]"
        
        # Visual progress bar (20 chars)
        bar_len = int(round(prob * 20))
        bar = "#" * bar_len + "-" * (20 - bar_len)
        
        print(f"{col:<22} | {prob:.4f}       | {status:<12} | [{bar}]")
    print("=" * 65 + "\n")


def test_custom_report(text: str):
    print("\n" + "-" * 65)
    print(f"Input Report: \"{text}\"")
    print("-" * 65)

    # 1. Show parsed clauses and negation detection
    clauses = split_clauses(text)
    print("\n[Clause Segmentation & Negation Analysis]:")
    for i, c in enumerate(clauses):
        neg = is_negated(c)
        neg_tag = "[NEGATED]" if neg else "[AFFIRMATIVE]"
        print(f"  {i+1}. {neg_tag:13s} : \"{c}\"")

    # 2. Extract NLP probabilities
    nlp_scores = extract_study_probabilities(text)
    print_score_table(nlp_scores, title=f"CLINICAL NLP PREDICTIONS ({len(text.split())} words)")


def test_dataset_study(index_or_uid: str):
    train_df = pd.read_csv("data/train.csv")
    
    if index_or_uid.isdigit():
        idx = int(index_or_uid)
        if idx < 0 or idx >= len(train_df):
            print(f"[ERROR] Index out of range (0 to {len(train_df)-1})")
            return
        row = train_df.iloc[idx]
    else:
        matches = train_df[train_df['StudyInstanceUID'].str.contains(index_or_uid)]
        if matches.empty:
            print(f"[ERROR] No study matching UID: {index_or_uid}")
            return
        row = matches.iloc[0]

    study_id = row['StudyInstanceUID']
    report = row['Report']
    
    print("\n" + "=" * 65)
    print(f"Study UID: {study_id}")
    print("=" * 65)
    print(f"Report: \"{str(report)[:300]}...\"\n")

    nlp_scores = extract_study_probabilities(report)
    print_score_table(nlp_scores, title=f"STUDY PREDICTIONS ({study_id[:20]}...)")


def interactive_menu():
    print("=" * 70)
    print("[MANUAL TEST] RSNA KNEE MULTIMODAL AI - INTERACTIVE TEST TOOL")
    print("=" * 70)
    print("Choose an option:")
    print("  1. Type or paste your own custom radiology report text")
    print("  2. Test a sample pre-set clinical case (ACL + Effusion)")
    print("  3. Test a sample pre-set multilingual case (Spanish / French / Dutch)")
    print("  4. Look up any study index from dataset (0 to 4406)")
    print("  5. Exit")
    print("=" * 70)

    while True:
        try:
            choice = input("\nEnter choice (1-5): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if choice == '1':
            print("\nPaste your report text (or type below and press Enter):")
            user_text = input("> ").strip()
            if user_text:
                test_custom_report(user_text)
            else:
                print("No text provided.")

        elif choice == '2':
            sample = "Impresion: Rotura completa de ligamento cruzado anterior con derrame articular abundante y contusion osea del condilo femoral externo."
            test_custom_report(sample)

        elif choice == '3':
            sample_fr = "Conclusion: Rupture du ligament collateral medial et lesion degenerative du menisque medial. Absence de fracture."
            test_custom_report(sample_fr)

        elif choice == '4':
            idx = input("Enter study index (0 - 4406): ").strip()
            test_dataset_study(idx)

        elif choice == '5' or choice.lower() in ['exit', 'quit', 'q']:
            print("Exiting manual testing tool. Done!")
            break
        else:
            print("Invalid choice, please enter 1, 2, 3, 4, or 5.")


if __name__ == '__main__':
    # If arguments passed from command line
    if len(sys.argv) > 1:
        custom_input = " ".join(sys.argv[1:])
        test_custom_report(custom_input)
    else:
        interactive_menu()
