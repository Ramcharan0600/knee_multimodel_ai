import re
import unicodedata
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

DATA_DIR = Path('data')
OUTPUT_DIR = Path('outputs/nlp_eval')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFD', text)
    text = "".join([c for c in text if unicodedata.category(c) != 'Mn'])
    text = text.lower()
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_clauses(text: str):
    normalized = normalize_text(text)
    clauses = re.split(r'[\.\;\n\r\>\•\*\-]|\d+\.\s*', normalized)
    return [c.strip() for c in clauses if len(c.strip()) > 1]

NEG_TERMS = [
    r'\bno\b', r'\bnot\b', r'\bsin\b', r'\bpas de\b', r'\bgeen\b', r'\bniet\b', r'\bzonder\b',
    r'\baucun\b', r'\bintact\b', r'\bnormal\b', r'\bconservad', r'\bunremarkable\b',
    r'\bpreserved\b', r'\blimite normal\b', r'\blimites normales\b', r'\bnegative\b',
    r'\bfree of\b', r'\bsans signe\b', r'\bsans lesion\b', r'\bintegre\b', r'\bkein\b',
    r'\bno tear\b', r'\bwithout tear\b', r'\bpas de dechirure\b', r'\bgeen scheur\b'
]
neg_re = re.compile('|'.join(NEG_TERMS))

def is_negated(clause: str) -> bool:
    return bool(neg_re.search(clause))

def extract_study_probabilities(report_text: str) -> dict:
    scores = {col: 0.05 for col in TARGET_COLS}
    if not isinstance(report_text, str) or len(report_text.strip()) == 0:
        return scores
        
    text_norm = normalize_text(report_text)
    clauses = split_clauses(report_text)
    
    impression_text = ""
    for kw in ['impresion:', 'conclusion:', 'impression:']:
        if kw in text_norm:
            impression_text = text_norm[text_norm.index(kw):]
            break
            
    # 1. Global Multicompartment Patterns
    if any(k in text_norm for k in ['tricompartmental', 'tri-compartmental', 'tricompartimental', 'generalized osteoarthritis', 'artrosis tricompartimental', 'pan-compartmental', 'three compartment']):
        if not any(is_negated(c) and 'tricompart' in c for c in clauses):
            scores['Medial OA'] = max(scores['Medial OA'], 0.90)
            scores['Lateral OA'] = max(scores['Lateral OA'], 0.90)
            scores['PF OA'] = max(scores['PF OA'], 0.90)
            
    if any(k in text_norm for k in ['femorotibial osteoarthritis', 'artrosis femorotibial', 'femorotibiaal kraak', 'bicompartmental', 'gonartrosis']):
        if not any(is_negated(c) and 'femorotibial' in c for c in clauses):
            scores['Medial OA'] = max(scores['Medial OA'], 0.85)
            scores['Lateral OA'] = max(scores['Lateral OA'], 0.85)

    if any(k in text_norm for k in ['both menisci', 'medial and lateral meniscus', 'menisco interno y lateral', 'meniscal tears', 'meniscopatia bilateral']):
        scores['Medial Meniscus'] = max(scores['Medial Meniscus'], 0.90)
        scores['Lateral Meniscus'] = max(scores['Lateral Meniscus'], 0.90)

    # 2. Detailed Clause-Level Scanning
    for c in clauses:
        neg = is_negated(c)
        
        # --- ACL ---
        if any(k in c for k in ['acl', 'anterior cruciate', 'cruzado anterior', 'croise anterieur', 'vkr', 'lca', 'voorste kruisband']):
            if neg:
                scores['ACL'] = min(scores['ACL'], 0.05)
            elif any(p in c for p in ['tear', 'torn', 'ruptur', 'rotura', 'dechirur', 'avulsion', 'sprain', 'injury', 'laxity', 'disrupt', 'complete', 'incompetent', 'lesion']):
                scores['ACL'] = 0.95 if c in impression_text else 0.85

        # --- MCL ---
        if any(k in c for k in ['mcl', 'medial collateral', 'colateral medial', 'colateral interno', 'collateral medial', 'mediale band', 'lcm']):
            if neg:
                scores['MCL'] = min(scores['MCL'], 0.05)
            elif any(p in c for p in ['tear', 'torn', 'ruptur', 'rotura', 'dechirur', 'sprain', 'entorse', 'esguince', 'thickening', 'edema', 'injury', 'strain', 'desgarro']):
                scores['MCL'] = 0.95 if c in impression_text else 0.85

        # --- Medial Meniscus ---
        if any(k in c for k in ['medial meniscus', 'menisco medial', 'menisco interno', 'menisque medial', 'menisque interne', 'mediale meniscus', 'binnenmeniskus', 'medial compartment']):
            if neg:
                scores['Medial Meniscus'] = min(scores['Medial Meniscus'], 0.05)
            elif any(p in c for p in ['tear', 'torn', 'rotura', 'ruptur', 'dechirur', 'fissur', 'scheur', 'extrusion', 'extruid', 'amputac', 'degen', 'complex', 'horizontal', 'radial', 'vertical', 'root', 'bucket handle', 'meniscopatia', 'fissuration']):
                scores['Medial Meniscus'] = 0.95 if c in impression_text else 0.88

        # --- Lateral Meniscus ---
        if any(k in c for k in ['lateral meniscus', 'menisco lateral', 'menisco externo', 'menisque lateral', 'menisque externe', 'laterale meniscus', 'aussenmeniskus', 'lateral compartment']):
            if neg:
                scores['Lateral Meniscus'] = min(scores['Lateral Meniscus'], 0.05)
            elif any(p in c for p in ['tear', 'torn', 'rotura', 'ruptur', 'dechirur', 'fissur', 'scheur', 'extrusion', 'extruid', 'amputac', 'degen', 'complex', 'horizontal', 'radial', 'vertical', 'discoid', 'meniscopatia', 'fissuration']):
                scores['Lateral Meniscus'] = 0.95 if c in impression_text else 0.88

        # --- Medial OA ---
        if any(k in c for k in ['medial', 'interno', 'interne']) and any(k in c for k in ['osteoarthr', 'artrosis', 'arthrose', 'chondropat', 'condropat', 'cartilage', 'chondral', 'joint space', 'compartment', 'condilo femoral medial', 'femorotibial', 'platillo tibial medial']):
            if neg:
                scores['Medial OA'] = min(scores['Medial OA'], 0.05)
            elif any(p in c for p in ['osteoarthr', 'artrosis', 'arthrose', 'chondropat', 'condropat', 'thinning', 'loss', 'defect', 'narrowing', 'pinzamiento', 'fissur', 'grade 2', 'grade 3', 'grade 4', 'grado 2', 'grado 3', 'grado 4', 'severe', 'moderate', 'ulcer', 'eburnation', 'sclerosis']):
                scores['Medial OA'] = 0.90 if c in impression_text else 0.80

        # --- Lateral OA ---
        if any(k in c for k in ['lateral', 'externo', 'externe']) and any(k in c for k in ['osteoarthr', 'artrosis', 'arthrose', 'chondropat', 'condropat', 'cartilage', 'chondral', 'joint space', 'compartment', 'condilo femoral lateral', 'femorotibial', 'platillo tibial lateral']):
            if neg:
                scores['Lateral OA'] = min(scores['Lateral OA'], 0.05)
            elif any(p in c for p in ['osteoarthr', 'artrosis', 'arthrose', 'chondropat', 'condropat', 'thinning', 'loss', 'defect', 'narrowing', 'pinzamiento', 'fissur', 'grade 2', 'grade 3', 'grade 4', 'grado 2', 'grado 3', 'grado 4', 'severe', 'moderate', 'ulcer', 'eburnation', 'sclerosis']):
                scores['Lateral OA'] = 0.90 if c in impression_text else 0.80

        # --- PF OA ---
        if any(k in c for k in ['patellofemoral', 'femoropatelar', 'retropatellar', 'retrorotulian', 'rotulian', 'patella', 'trochle', 'troclea', 'faceta rotuliana', 'patellar cartilage']):
            if neg:
                scores['PF OA'] = min(scores['PF OA'], 0.05)
            elif any(p in c for p in ['osteoarthr', 'artrosis', 'arthrose', 'chondropat', 'condropat', 'chondromalac', 'condromalac', 'thinning', 'loss', 'defect', 'narrowing', 'fissur', 'grade', 'grado', 'severe', 'moderate', 'ulcer']):
                scores['PF OA'] = 0.90 if c in impression_text else 0.80

        # --- Effusion ---
        if any(k in c for k in ['effusion', 'derrame', 'epanchement', 'hydrops', 'joint fluid', 'liquido articular', 'suprapatellar bursa', 'suprapatellar fluid', 'fluid accumulation', 'fluid in the', 'synovial fluid']):
            if neg:
                scores['Effusion'] = min(scores['Effusion'], 0.05)
            elif any(p in c for p in ['effusion', 'derrame', 'epanchement', 'present', 'mild', 'moderate', 'large', 'massive', 'abundant', 'small', 'noted', 'fluid', 'accumulat', 'distension', 'bursitis', 'trace']):
                scores['Effusion'] = 0.92 if c in impression_text else 0.85

        # --- Synovitis ---
        if any(k in c for k in ['synovit', 'sinovit', 'synovial', 'sinovial', 'hoffit', 'synovialis']):
            if neg:
                scores['Synovitis'] = min(scores['Synovitis'], 0.05)
            elif any(p in c for p in ['synovit', 'sinovit', 'thickening', 'engrosamiento', 'epaissement', 'inflammat', 'proliferation', 'proliferacion', 'hoffitis', 'hypertroph', 'hyperplasia']):
                scores['Synovitis'] = 0.92 if c in impression_text else 0.85

        # --- Baker's Cyst ---
        if any(k in c for k in ['baker', 'popliteal cyst', 'quiste popliteo', 'kyste de baker', 'kyste poplite', 'popliteale cyste', 'quiste de baker', 'gastrocnemius-semimembranosus', 'popliteal collection']):
            if neg:
                scores['Baker\'s'] = min(scores['Baker\'s'], 0.05)
            elif any(p in c for p in ['baker', 'cyst', 'quiste', 'kyste', 'cyste', 'present', 'noted', 'measur', 'collection', 'distension']):
                scores['Baker\'s'] = 0.95 if c in impression_text else 0.88

        # --- Contusion ---
        if any(k in c for k in ['contusion', 'bone bruise', 'bone marrow edema', 'edema oseo', 'edema medular', 'edema trabecular', 'edeme osseux', 'knochenmarkodem', 'marrow edema', 'marrow contusion', 'subchondral edema']):
            if neg:
                scores['Contusion'] = min(scores['Contusion'], 0.05)
            elif any(p in c for p in ['contusion', 'bruise', 'edema', 'edeme', 'present', 'noted', 'trabecular', 'subchondral', 'reactive', 'kissing']):
                scores['Contusion'] = 0.90 if c in impression_text else 0.82

        # --- Fracture ---
        if any(k in c for k in ['fractur', 'fractura', 'fracture', 'cortical breach', 'cortical step', 'frakt', 'avulsion fracture', 'microfractur']):
            if neg:
                scores['Fracture'] = min(scores['Fracture'], 0.05)
            elif any(p in c for p in ['fractur', 'fractura', 'fracture', 'insufficiency', 'avulsion', 'step', 'depression', 'disrupt', 'impacted', 'collapse']):
                scores['Fracture'] = 0.95 if c in impression_text else 0.88

    return scores

def run_extraction_and_benchmarks():
    print('=== Running Multilingual Report NLP Extractor & Benchmark ===\n')
    
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    labeled_df = train_df[train_df['ACL'].notna()].copy()
    
    # Benchmark on Gold Standard (N=58)
    gold_preds = [extract_study_probabilities(r) for r in labeled_df['Report']]
    gold_pred_df = pd.DataFrame(gold_preds)
    
    metrics = []
    for target in TARGET_COLS:
        y_true = labeled_df[target].values.astype(int)
        y_prob = gold_pred_df[target].values
        y_bin = (y_prob >= 0.5).astype(int)
        
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 1.0
        f1 = f1_score(y_true, y_bin, zero_division=0)
        prec = precision_score(y_true, y_bin, zero_division=0)
        rec = recall_score(y_true, y_bin, zero_division=0)
        
        metrics.append({
            'Pathology': target,
            'Gold Positives': int(np.sum(y_true)),
            'Pred Positives': int(np.sum(y_bin)),
            'Precision': round(prec, 3),
            'Recall': round(rec, 3),
            'F1-Score': round(f1, 3),
            'AUC-ROC': round(auc, 3)
        })
        
    metrics_df = pd.DataFrame(metrics)
    print("=== NLP Extractor Benchmark on Gold Standard (N=58) ===")
    print(metrics_df.to_string(index=False))
    
    macro_auc = metrics_df['AUC-ROC'].mean()
    macro_f1 = metrics_df['F1-Score'].mean()
    print(f"\n[Result] Macro-Averaged AUC-ROC: {macro_auc:.4f}")
    print(f"[Result] Macro-Averaged F1-Score: {macro_f1:.4f}")
    
    metrics_df.to_csv(OUTPUT_DIR / 'nlp_benchmark_metrics.csv', index=False)
    
    # 3. Generate Pseudo-Labels for ALL 4,407 Studies
    print('\nGenerating Pseudo-Labels for Full Cohort (N=4,407)...')
    all_preds = [extract_study_probabilities(r) for r in train_df['Report']]
    pseudo_df = pd.DataFrame(all_preds)
    
    out_df = train_df[['StudyInstanceUID', 'Report']].copy()
    for col in TARGET_COLS:
        out_df[f'{col}_prob'] = pseudo_df[col].values
        out_df[col] = train_df[col].fillna(pseudo_df[col])
        out_df[f'{col}_hard'] = (out_df[col] >= 0.5).astype(int)
        
    out_df.to_csv(DATA_DIR / 'train_pseudo_labeled.csv', index=False)
    print(f'Full pseudo-labeled dataset saved to: {DATA_DIR / "train_pseudo_labeled.csv"}')
    print(f'Total annotated studies now available: {len(out_df):,}')

if __name__ == '__main__':
    run_extraction_and_benchmarks()
