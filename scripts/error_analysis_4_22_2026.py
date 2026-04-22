#!/usr/bin/env python3
"""
Phase 5 Error Analysis Script
==============================
Comprehensive error analysis for dependency parsing across Creole languages.

Features:
- Root attachment error tracking (HEAD=0)
- Relation class grouping (core args, modifiers, function words)
- Head distance decile analysis (local vs long-range dependencies)
- Per-language error rates (for multilingual data)
- Function vs content word distinction
- Confusion matrix with signal filtering

Usage:
    python3 phase5_error_analysis.py <gold_file> <pred_file> <output_dir> [--language HC|MC]

Output:
    {output_dir}/
    ├── error_by_relation_class.csv
    ├── error_by_head_distance.csv
    ├── error_by_deprel.csv
    ├── confusion_matrix_filtered.csv
    ├── analysis_summary.txt
    └── visualizations/ (PNG plots)
"""

import pandas as pd
import numpy as np
import argparse
import os
from collections import defaultdict, Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ============================================================================
# CONFIGURATION
# ============================================================================

FUNCTION_TAGS = {"DET", "ADP", "AUX", "PART", "SCONJ", "CCONJ"}

# Dependency relation groupings (UD standard)
RELATION_CLASSES = {
    'core_args': {'nsubj', 'obj', 'iobj', 'csubj', 'ccomp'},
    'modifiers': {'nmod', 'amod', 'acl', 'advcl', 'obl'},
    'function': {'case', 'mark', 'det', 'aux', 'cop', 'fixed'},
    'coordination': {'conj', 'cc'},
    'compound': {'compound', 'flat'},
    'other': set()
}

# Ensure 'other' captures uncategorized relations
ALL_RELATIONS = set()
for rel_list in RELATION_CLASSES.values():
    if rel_list:
        ALL_RELATIONS.update(rel_list)


# ============================================================================
# CORE READING & PARSING
# ============================================================================

def read_conllu(filepath):
    """Read CoNLL-U file and return list of sentences (each sentence = list of tokens)."""
    sentences = []
    sentence = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                if sentence:
                    sentences.append(sentence)
                    sentence = []
                continue
            columns = line.split('\t')
            if len(columns) >= 10:
                sentence.append(columns)
        if sentence:
            sentences.append(sentence)
    return sentences


def compare_predictions(gold_sentences, pred_sentences):
    """Compare gold vs predicted; return DataFrame with error analysis."""
    results = []
    
    for sent_idx, (gold, pred) in enumerate(zip(gold_sentences, pred_sentences)):
        # Build dictionaries keyed by token ID for robust alignment
        gold_tokens = {}
        for tok in gold:
            if '-' not in tok[0] and '.' not in tok[0]:
                try:
                    token_id = int(tok[0])
                    gold_tokens[token_id] = tok
                except ValueError:
                    pass
        
        pred_tokens = {}
        for tok in pred:
            if '-' not in tok[0] and '.' not in tok[0]:
                try:
                    token_id = int(tok[0])
                    pred_tokens[token_id] = tok
                except ValueError:
                    pass
        
        # Count valid tokens only
        sent_length = len(gold_tokens)
        
        # Process only tokens that exist in both gold and pred
        for token_id in sorted(gold_tokens.keys()):
            if token_id not in pred_tokens:
                continue  # Skip if prediction missing for this token
            
            g_token = gold_tokens[token_id]
            p_token = pred_tokens[token_id]
            
            form = g_token[1] if len(g_token) > 1 else ""
            upos = g_token[3] if len(g_token) > 3 else None
            
            # Gold values
            gold_head_str = g_token[6] if len(g_token) > 6 else None
            gold_deprel = g_token[7] if len(g_token) > 7 else None
            
            # Predicted values
            pred_head_str = p_token[6] if len(p_token) > 6 else None
            pred_deprel = p_token[7] if len(p_token) > 7 else None
            
            # Convert head to int
            try:
                gold_head = int(gold_head_str) if gold_head_str else None
            except ValueError:
                gold_head = None
            
            try:
                pred_head = int(pred_head_str) if pred_head_str else None
            except ValueError:
                pred_head = None
            
            # Head distance (0 for root, otherwise absolute difference)
            head_distance = None
            if gold_head is not None and token_id is not None:
                if gold_head == 0:
                    head_distance = 0  # Root has distance 0
                else:
                    head_distance = abs(token_id - gold_head)
            
            # Correctness flags
            correct_head = 1 if gold_head == pred_head else 0
            correct_deprel = 1 if gold_deprel == pred_deprel else 0
            
            # Root vs non-root
            is_root = 1 if gold_head == 0 else 0
            root_correct = 1 if is_root and correct_head else 0
            
            # Function word classification
            is_function = 1 if upos in FUNCTION_TAGS else 0 if upos is not None else None
            
            # Categorize relation
            relation_class = None
            if gold_deprel:
                for rel_class, rel_set in RELATION_CLASSES.items():
                    if gold_deprel in rel_set:
                        relation_class = rel_class
                        break
                if not relation_class:
                    relation_class = 'other'
            
            results.append({
                'Sentence_ID': sent_idx,
                'Token_ID': token_id,
                'Token': form,
                'Sentence_Length': sent_length,
                'Token_Index': token_id,  # Use actual token ID
                'UPOS': upos,
                'Is_Function': is_function,
                'Gold_Head': gold_head,
                'Pred_Head': pred_head,
                'Head_Distance': head_distance,
                'Gold_Deprel': gold_deprel,
                'Pred_Deprel': pred_deprel,
                'Correct_Head': correct_head,
                'Correct_Deprel': correct_deprel,
                'Is_Root': is_root,
                'Root_Correct': root_correct if is_root else None,
                'Relation_Class': relation_class
            })
    
    return pd.DataFrame(results)


# ============================================================================
# ERROR ANALYSIS BY CATEGORY
# ============================================================================

def analyze_errors_by_deprel(df):
    """Compute error rates per dependency relation."""
    deprel_errors = {}
    unique_deprels = df['Gold_Deprel'].dropna().unique()
    
    for deprel in sorted(unique_deprels):
        subset = df[df['Gold_Deprel'] == deprel]
        error_rate = 1 - subset['Correct_Deprel'].mean()
        count = len(subset)
        deprel_errors[deprel] = {
            'count': count,
            'error_rate': error_rate,
            'head_errors': 1 - subset['Correct_Head'].mean(),
            'deprel_errors': error_rate
        }
    
    return pd.DataFrame(deprel_errors).T.sort_values('count', ascending=False)


def analyze_errors_by_relation_class(df):
    """Compute error rates per relation class (core args, modifiers, etc.)."""
    class_errors = {}
    
    for rel_class in RELATION_CLASSES.keys():
        subset = df[df['Relation_Class'] == rel_class]
        if len(subset) == 0:
            continue
        
        class_errors[rel_class] = {
            'count': len(subset),
            'error_rate': 1 - subset['Correct_Deprel'].mean(),
            'head_errors': 1 - subset['Correct_Head'].mean(),
            'avg_distance': subset['Head_Distance'].mean(),
            'avg_length': subset['Sentence_Length'].mean()
        }
    
    return pd.DataFrame(class_errors).T


def analyze_errors_by_head_distance(df):
    """Compute error rates by head-to-token distance (local vs long-range)."""
    # Define distance bins: non-overlapping, with clear naming for paper
    distance_bins = {
        '0': (0, 1),
        '1': (1, 2),
        '2': (2, 3),
        '3-5': (3, 6),
        '6-10': (6, 11),
        '11-50': (11, 51),
        '50+': (51, 10000)
    }
    
    distance_errors = {}
    for bin_name, (min_dist, max_dist) in distance_bins.items():
        subset = df[(df['Head_Distance'] >= min_dist) & (df['Head_Distance'] < max_dist)]
        if len(subset) == 0:
            continue
        
        distance_errors[bin_name] = {
            'count': len(subset),
            'error_rate': 1 - subset['Correct_Deprel'].mean(),
            'head_errors': 1 - subset['Correct_Head'].mean(),
            'avg_sent_len': subset['Sentence_Length'].mean()
        }
    
    return pd.DataFrame(distance_errors).T


def analyze_root_errors(df):
    """Analyze ROOT (HEAD=0) attachment separately."""
    root_tokens = df[df['Is_Root'] == 1]
    non_root_tokens = df[df['Is_Root'] == 0]
    
    root_error_rate = 1 - root_tokens['Correct_Head'].mean() if len(root_tokens) > 0 else None
    non_root_error_rate = 1 - non_root_tokens['Correct_Head'].mean()
    
    return {
        'root_count': len(root_tokens),
        'root_error_rate': root_error_rate,
        'non_root_count': len(non_root_tokens),
        'non_root_error_rate': non_root_error_rate,
        'root_vs_nonroot_diff': (root_error_rate - non_root_error_rate) if root_error_rate else None
    }


def analyze_by_function_word(df):
    """Compare error rates for function vs content words."""
    function_df = df[df['Is_Function'] == 1]
    content_df = df[df['Is_Function'] == 0]
    
    return {
        'function_word_count': len(function_df),
        'function_error_rate': 1 - function_df['Correct_Deprel'].mean() if len(function_df) > 0 else None,
        'content_word_count': len(content_df),
        'content_error_rate': 1 - content_df['Correct_Deprel'].mean() if len(content_df) > 0 else None
    }


def analyze_by_sentence_length(df, n_bins=5):
    """Compute error rates by sentence length (finer granularity)."""
    length_bins = pd.qcut(df['Sentence_Length'], q=n_bins, duplicates='drop')
    length_errors = {}
    
    for bin_label in length_bins.cat.categories:
        subset = df[length_bins == bin_label]
        if len(subset) == 0:
            continue
        
        length_errors[f"length_{int(bin_label.left)}-{int(bin_label.right)}"] = {
            'count': len(subset),
            'error_rate': 1 - subset['Correct_Deprel'].mean(),
            'head_errors': 1 - subset['Correct_Head'].mean()
        }
    
    return pd.DataFrame(length_errors).T


# ============================================================================
# CONFUSION MATRIX & SIGNAL FILTERING
# ============================================================================

def get_confusion_matrix(df):
    """Compute confusion matrix and filter for significant signals."""
    # Clean data: remove None/NaN values
    df_cm = df.dropna(subset=['Gold_Deprel', 'Pred_Deprel'])
    y_true = df_cm['Gold_Deprel']
    y_pred = df_cm['Pred_Deprel']
    
    unique_labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    cm_df = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)
    
    # Filter confusions: keep only those >3× random chance and with ≥5 occurrences
    total = cm.sum()
    significant_confusions = []
    
    for i, gold in enumerate(unique_labels):
        for j, pred in enumerate(unique_labels):
            count = cm[i][j]
            if gold != pred and count > 0:
                # Baseline chance
                gold_freq = cm[i, :].sum()
                pred_freq = cm[:, j].sum()
                chance = (gold_freq * pred_freq) / total
                
                if count >= 5 and count > 3 * chance:
                    significant_confusions.append({
                        'Gold_Label': gold,
                        'Pred_Label': pred,
                        'Count': count,
                        'Chance': chance,
                        'Ratio': count / chance if chance > 0 else 0
                    })
    
    # Return confusion matrix and significant pairs
    if significant_confusions:
        return cm_df, pd.DataFrame(significant_confusions).sort_values('Count', ascending=False)
    else:
        return cm_df, pd.DataFrame(columns=['Gold_Label', 'Pred_Label', 'Count', 'Chance', 'Ratio'])


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_error_rates(df_errors, title, filename):
    """Plot error rates as bar chart."""
    plt.figure(figsize=(12, 6))
    x = range(len(df_errors))
    y = df_errors['error_rate'].values
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(df_errors)))
    
    plt.bar(x, y, color=colors)
    plt.axhline(y=df_errors['error_rate'].mean(), color='red', linestyle='--', label='Mean')
    plt.xticks(x, df_errors.index, rotation=45, ha='right')
    plt.ylabel('Error Rate')
    plt.title(title)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_head_distance_trend(df_errors, filename):
    """Plot error rate trend by head distance."""
    plt.figure(figsize=(12, 6))
    x_labels = df_errors.index.tolist()
    y = df_errors['error_rate'].values
    
    plt.plot(range(len(y)), y, marker='o', linewidth=2, markersize=8, color='steelblue')
    plt.fill_between(range(len(y)), y, alpha=0.3, color='steelblue')
    plt.xticks(range(len(y)), x_labels, rotation=45, ha='right')
    plt.ylabel('Error Rate')
    plt.xlabel('Head Distance')
    plt.title('Error Rate Trend by Head Distance (Local → Long-Range)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm_df, filename, top_n=20):
    """Plot confusion matrix heatmap (top N×N labels by frequency)."""
    # Keep top N most frequent labels
    row_sums = cm_df.sum(axis=1)
    col_sums = cm_df.sum(axis=0)
    freq = row_sums + col_sums
    top_labels = freq.nlargest(top_n).index
    
    cm_subset = cm_df.loc[top_labels, top_labels]
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_subset, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.xlabel('Predicted Label')
    plt.ylabel('Gold Label')
    plt.title('Confusion Matrix (Top 20 Labels)')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN ANALYSIS & REPORTING
# ============================================================================

def generate_report(df, output_dir, language=''):
    """Generate comprehensive error analysis report."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(f"PHASE 5 ERROR ANALYSIS — {language if language else 'All Languages'}")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # ---- ROOT ATTACHMENT ANALYSIS ----
    report_lines.append("ROOT ATTACHMENT ANALYSIS")
    report_lines.append("-" * 80)
    root_stats = analyze_root_errors(df)
    report_lines.append(f"Root tokens (HEAD=0): {root_stats['root_count']}")
    report_lines.append(f"  Error rate: {root_stats['root_error_rate']:.2%}")
    report_lines.append(f"Non-root tokens: {root_stats['non_root_count']}")
    report_lines.append(f"  Error rate: {root_stats['non_root_error_rate']:.2%}")
    if root_stats['root_vs_nonroot_diff'] is not None:
        report_lines.append(f"Difference (root - non-root): {root_stats['root_vs_nonroot_diff']:+.2%}")
    report_lines.append("")
    
    # ---- RELATION CLASS ANALYSIS ----
    report_lines.append("RELATION CLASS ERROR RATES")
    report_lines.append("-" * 80)
    df_class = analyze_errors_by_relation_class(df)
    report_lines.append(df_class.to_string())
    report_lines.append("")
    
    df_class.to_csv(os.path.join(output_dir, 'error_by_relation_class.csv'))
    plot_error_rates(df_class, 'Error Rate by Relation Class', 
                     os.path.join(output_dir, 'visualizations', 'error_by_relation_class.png'))
    
    # ---- HEAD DISTANCE ANALYSIS ----
    report_lines.append("HEAD DISTANCE ANALYSIS")
    report_lines.append("-" * 80)
    df_distance = analyze_errors_by_head_distance(df)
    report_lines.append(df_distance.to_string())
    report_lines.append("")
    
    df_distance.to_csv(os.path.join(output_dir, 'error_by_head_distance.csv'))
    plot_head_distance_trend(df_distance, 
                             os.path.join(output_dir, 'visualizations', 'error_by_head_distance.png'))
    
    # ---- PER-DEPREL ANALYSIS ----
    report_lines.append("PER-DEPENDENCY-RELATION ERROR RATES")
    report_lines.append("-" * 80)
    df_deprel = analyze_errors_by_deprel(df)
    report_lines.append(df_deprel.to_string())
    report_lines.append("")
    
    df_deprel.to_csv(os.path.join(output_dir, 'error_by_deprel.csv'))
    plot_error_rates(df_deprel.head(15), 'Top 15 Dependency Relations by Frequency',
                     os.path.join(output_dir, 'visualizations', 'error_by_top_deprels.png'))
    
    # ---- FUNCTION VS CONTENT WORD ANALYSIS ----
    report_lines.append("FUNCTION VS CONTENT WORD ANALYSIS")
    report_lines.append("-" * 80)
    func_stats = analyze_by_function_word(df)
    report_lines.append(f"Function words: {func_stats['function_word_count']}")
    report_lines.append(f"  Error rate: {func_stats['function_error_rate']:.2%}" if func_stats['function_error_rate'] else "  N/A")
    report_lines.append(f"Content words: {func_stats['content_word_count']}")
    report_lines.append(f"  Error rate: {func_stats['content_error_rate']:.2%}" if func_stats['content_error_rate'] else "  N/A")
    report_lines.append("")
    
    # ---- SENTENCE LENGTH ANALYSIS ----
    report_lines.append("SENTENCE LENGTH ANALYSIS (5 Quintiles)")
    report_lines.append("-" * 80)
    df_length = analyze_by_sentence_length(df, n_bins=5)
    report_lines.append(df_length.to_string())
    report_lines.append("")
    
    df_length.to_csv(os.path.join(output_dir, 'error_by_sentence_length.csv'))
    
    # ---- CONFUSION MATRIX ----
    report_lines.append("CONFUSION MATRIX (Significant Confusions >3× Chance, ≥5 occurrences)")
    report_lines.append("-" * 80)
    cm_df, sig_confusions = get_confusion_matrix(df)
    
    report_lines.append(f"Total unique relations: {len(cm_df)}")
    report_lines.append(f"Significant confusion pairs (>3× baseline, ≥5 count): {len(sig_confusions)}")
    report_lines.append("")
    if len(sig_confusions) > 0:
        report_lines.append("Top 20 Significant Confusion Pairs:")
        report_lines.append(sig_confusions.head(20).to_string(index=False))
    report_lines.append("")
    
    cm_df.to_csv(os.path.join(output_dir, 'confusion_matrix.csv'))
    sig_confusions.to_csv(os.path.join(output_dir, 'confusion_matrix_filtered.csv'), index=False)
    plot_confusion_matrix(cm_df, os.path.join(output_dir, 'visualizations', 'confusion_matrix.png'))
    
    # ---- SUMMARY STATISTICS ----
    report_lines.append("OVERALL STATISTICS")
    report_lines.append("-" * 80)
    report_lines.append(f"Total tokens analyzed: {len(df)}")
    report_lines.append(f"Head accuracy (UAS): {df['Correct_Head'].mean():.2%}")
    report_lines.append(f"Deprel accuracy (LAS): {df['Correct_Deprel'].mean():.2%}")
    
    # Deprel-only accuracy given correct head (with safety check)
    subset_correct = df[df['Correct_Head'] == 1]
    if len(subset_correct) > 0:
        dep_given_head = subset_correct['Correct_Deprel'].mean()
        report_lines.append(f"Deprel-only accuracy (given correct head): {dep_given_head:.2%}")
    else:
        report_lines.append(f"Deprel-only accuracy (given correct head): N/A (no correct heads)")
    report_lines.append("")
    
    # Write report
    report_text = "\n".join(report_lines)
    print(report_text)
    
    with open(os.path.join(output_dir, 'analysis_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 5 Error Analysis for Dependency Parsing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 phase5_error_analysis.py gold.conllu pred.conllu ./results --language HC
  python3 phase5_error_analysis.py data/mc_test.conllu predictions/mc_seed1.conllu ./mc_results
        """
    )
    parser.add_argument("gold", help="Path to gold CoNLL-U file")
    parser.add_argument("pred", help="Path to predicted CoNLL-U file")
    parser.add_argument("output_dir", help="Output directory for results")
    parser.add_argument("--language", default="", help="Language code (HC, MC, FR, etc.) for report header")
    
    args = parser.parse_args()
    
    print(f"Reading gold standard: {args.gold}")
    gold_sentences = read_conllu(args.gold)
    
    print(f"Reading predictions: {args.pred}")
    pred_sentences = read_conllu(args.pred)
    
    if len(gold_sentences) != len(pred_sentences):
        print(f"WARNING: Sentence count mismatch (gold: {len(gold_sentences)}, pred: {len(pred_sentences)})")
    
    print("Comparing predictions...")
    df = compare_predictions(gold_sentences, pred_sentences)
    
    print(f"Generating analysis report...")
    generate_report(df, args.output_dir, language=args.language)
    
    print(f"\nAnalysis complete. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
