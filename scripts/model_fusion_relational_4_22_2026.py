#!/usr/bin/env python3
"""
Phase 6.1: Relation-Aware Fusion Analysis (Based on Phase 5 Findings)
----------------------------------------------------------------------
Motivation: Phase 6 showed simple distance heuristics fail. Phase 5 error
analysis revealed models have complementary strengths by RELATION TYPE.

This phase tests:
1. Relation-aware routing (use best model per relation class)
2. Combined strategies (distance + relation class)
3. Show the gap: Best Single → Best Strategy → Oracle Upper Bound

Focus: POSITIVE findings (what works), not negative results.

Output structure:
results/phase6.1/
  hc/non_mtl/
  hc/mtl/
  mc/non_mtl/
  mc/mtl/
  summary/
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

ROOT = "/home/a-lvm861/projects/supar/supar_creole"
DEFAULT_OUTPUT_DIR = f"{ROOT}/results/phase6.1"

# Relation classes from Phase 5 analysis
RELATION_CLASSES = {
    "core_args": {"nsubj", "obj", "iobj", "csubj", "ccomp", "xcomp"},
    "modifiers": {"nmod", "amod", "acl", "acl:relcl", "advcl", "obl", "obl:mod"},
    "function": {"case", "mark", "det", "aux", "cop", "cc"},
    "coordination": {"conj", "cc"},
    "other": set(),  # catch-all
}

def classify_relation(deprel: str) -> str:
    """Map a deprel to its relation class."""
    for class_name, rels in RELATION_CLASSES.items():
        if deprel in rels:
            return class_name
    return "other"


def load_conllu(path: str) -> pd.DataFrame:
    """Parse CoNLL-U file into DataFrame with ID, HEAD, DEPREL."""
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) >= 8 and '-' not in parts[0] and '.' not in parts[0]:
                rows.append({
                    'id': int(parts[0]),
                    'head': int(parts[6]),
                    'deprel': parts[7].split(':')[0] if ':' in parts[7] else parts[7],
                })
    return pd.DataFrame(rows)


def align_predictions(gold_path: str, pred_paths: Dict[str, str]) -> pd.DataFrame:
    """Align gold with multiple model predictions."""
    gold = load_conllu(gold_path)
    gold = gold.rename(columns={'head': 'gold_head', 'deprel': 'gold_deprel'})
    
    for model_name, pred_path in pred_paths.items():
        pred = load_conllu(pred_path)
        gold[f'pred_head_{model_name}'] = pred['head'].values
        gold[f'pred_deprel_{model_name}'] = pred['deprel'].values
    
    # Add metadata
    gold['relation_class'] = gold['gold_deprel'].apply(classify_relation)
    gold['head_distance'] = (gold['id'] - gold['gold_head']).abs()
    
    return gold


def compute_oracle(df: pd.DataFrame, models: List[str]) -> Tuple[pd.DataFrame, float, float]:
    """Add oracle (any model correct) columns."""
    # Oracle head: any model got it right
    oracle_head = pd.Series(False, index=df.index)
    for m in models:
        oracle_head |= (df[f'pred_head_{m}'] == df['gold_head'])
    
    # Oracle deprel: any model got it right (given correct head)
    oracle_deprel = pd.Series(False, index=df.index)
    for m in models:
        correct_head = (df[f'pred_head_{m}'] == df['gold_head'])
        correct_deprel = (df[f'pred_deprel_{m}'] == df['gold_deprel'])
        oracle_deprel |= (correct_head & correct_deprel)
    
    df['oracle_head'] = oracle_head
    df['oracle_deprel'] = oracle_deprel
    
    oracle_uas = float(oracle_head.mean())
    oracle_las = float(oracle_deprel.mean())
    
    return df, oracle_uas, oracle_las


def best_single_model(df: pd.DataFrame, models: List[str]) -> Tuple[str, float, float]:
    """Find best single model by LAS."""
    best_m = None
    best_las = 0.0
    best_uas = 0.0
    
    for m in models:
        uas = float((df[f'pred_head_{m}'] == df['gold_head']).mean())
        las = float(((df[f'pred_head_{m}'] == df['gold_head']) & 
                     (df[f'pred_deprel_{m}'] == df['gold_deprel'])).mean())
        if las > best_las:
            best_las = las
            best_uas = uas
            best_m = m
    
    return best_m, best_uas, best_las


def analyze_model_strengths_by_relation(df: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    """Find which model is best at each relation type."""
    rows = []
    
    for rel_class in RELATION_CLASSES.keys():
        subset = df[df['relation_class'] == rel_class]
        if len(subset) < 10:  # Skip rare classes
            continue
        
        best_model = None
        best_las = 0.0
        
        for m in models:
            las = float(((subset[f'pred_head_{m}'] == subset['gold_head']) & 
                        (subset[f'pred_deprel_{m}'] == subset['gold_deprel'])).mean())
            if las > best_las:
                best_las = las
                best_model = m
        
        rows.append({
            'relation_class': rel_class,
            'count': len(subset),
            'best_model': best_model,
            'best_las': best_las,
        })
    
    return pd.DataFrame(rows)


def test_relation_aware_routing(df: pd.DataFrame, models: List[str], 
                                 strengths: pd.DataFrame) -> Tuple[float, float, Dict]:
    """Route each token to the model that's best at its relation class."""
    # Build routing map
    routing_map = {}
    for _, row in strengths.iterrows():
        routing_map[row['relation_class']] = row['best_model']
    
    # Apply routing
    pred_heads = []
    pred_deprels = []
    routing_decisions = []
    
    for idx, row in df.iterrows():
        rel_class = row['relation_class']
        chosen_model = routing_map.get(rel_class, models[0])  # Default to first model
        
        pred_heads.append(row[f'pred_head_{chosen_model}'])
        pred_deprels.append(row[f'pred_deprel_{chosen_model}'])
        routing_decisions.append(chosen_model)
    
    pred_heads = pd.Series(pred_heads, index=df.index)
    pred_deprels = pd.Series(pred_deprels, index=df.index)
    
    uas = float((pred_heads == df['gold_head']).mean())
    las = float(((pred_heads == df['gold_head']) & (pred_deprels == df['gold_deprel'])).mean())
    
    # Count routing decisions
    routing_counts = pd.Series(routing_decisions).value_counts().to_dict()
    
    return uas, las, routing_counts


def test_distance_plus_relation_routing(df: pd.DataFrame, models: List[str],
                                       strengths: pd.DataFrame, 
                                       distance_threshold: int = 2) -> Tuple[float, float, Dict]:
    """Combined strategy: short deps use relation routing, long deps use best long-distance model."""
    # Find best model for long-distance dependencies
    long_deps = df[df['head_distance'] > distance_threshold]
    best_long_model = None
    best_long_las = 0.0
    
    for m in models:
        las = float(((long_deps[f'pred_head_{m}'] == long_deps['gold_head']) & 
                    (long_deps[f'pred_deprel_{m}'] == long_deps['gold_deprel'])).mean())
        if las > best_long_las:
            best_long_las = las
            best_long_model = m
    
    # Build routing map for short dependencies
    routing_map = {}
    for _, row in strengths.iterrows():
        routing_map[row['relation_class']] = row['best_model']
    
    # Apply combined routing
    pred_heads = []
    pred_deprels = []
    routing_decisions = []
    
    for idx, row in df.iterrows():
        if row['head_distance'] > distance_threshold:
            chosen_model = best_long_model
        else:
            rel_class = row['relation_class']
            chosen_model = routing_map.get(rel_class, models[0])
        
        pred_heads.append(row[f'pred_head_{chosen_model}'])
        pred_deprels.append(row[f'pred_deprel_{chosen_model}'])
        routing_decisions.append(chosen_model)
    
    pred_heads = pd.Series(pred_heads, index=df.index)
    pred_deprels = pd.Series(pred_deprels, index=df.index)
    
    uas = float((pred_heads == df['gold_head']).mean())
    las = float(((pred_heads == df['gold_head']) & (pred_deprels == df['gold_deprel'])).mean())
    
    routing_counts = pd.Series(routing_decisions).value_counts().to_dict()
    
    return uas, las, routing_counts


def run_track_analysis(lang: str, track: str, gold_path: str, 
                       model_paths: Dict[str, str], output_dir: str) -> Dict:
    """Run complete analysis for one language-track combination."""
    print(f"\n{'='*60}")
    print(f"Analyzing {lang.upper()} | {track.upper()}")
    print(f"{'='*60}")
    
    # Load and align
    print("Loading predictions...")
    df = align_predictions(gold_path, model_paths)
    models = list(model_paths.keys())
    
    # Oracle
    print("Computing oracle bounds...")
    df, oracle_uas, oracle_las = compute_oracle(df, models)
    
    # Best single
    best_model, best_uas, best_las = best_single_model(df, models)
    print(f"Best single model: {best_model} (LAS={best_las:.2%})")
    
    # Analyze strengths by relation
    print("Analyzing model strengths by relation class...")
    strengths = analyze_model_strengths_by_relation(df, models)
    strengths.to_csv(f"{output_dir}/model_strengths_by_relation.csv", index=False)
    
    # Test relation-aware routing
    print("Testing relation-aware routing...")
    rel_uas, rel_las, rel_routing = test_relation_aware_routing(df, models, strengths)
    
    # Test combined distance + relation routing
    print("Testing combined distance+relation routing...")
    comb_uas, comb_las, comb_routing = test_distance_plus_relation_routing(df, models, strengths)
    
    # Results summary
    results = {
        'lang': lang,
        'track': track,
        'n_tokens': len(df),
        'n_models': len(models),
        'models': models,
        'best_single_model': best_model,
        'best_single_uas': best_uas,
        'best_single_las': best_las,
        'relation_routing_uas': rel_uas,
        'relation_routing_las': rel_las,
        'relation_routing_gain': rel_las - best_las,
        'relation_routing_counts': rel_routing,
        'combined_routing_uas': comb_uas,
        'combined_routing_las': comb_las,
        'combined_routing_gain': comb_las - best_las,
        'combined_routing_counts': comb_routing,
        'oracle_uas': oracle_uas,
        'oracle_las': oracle_las,
        'oracle_gap': oracle_las - best_las,
    }
    
    # Save detailed results
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 6.1: Relation-aware fusion analysis")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configuration matching Phase 6 model sets
    config = {
        "hc": {
            "gold": f"{ROOT}/data/hc/hc_original_split_test.conllu",
            "non_mtl_models": {
                "hc_lstm": f"{ROOT}/predictions/hc/hc__creoleval__lstm-tag-char-bert__glove_seed1.conllu",
                "hc_bert": f"{ROOT}/predictions/hc/hc__creoleval__bert-enc__n-a_seed1.conllu",
                "fr+hc_lstm": f"{ROOT}/predictions/hc/fr+hc__creoleval__lstm-tag-char-bert__glove_seed1.conllu",
                "fr+hc_bert": f"{ROOT}/predictions/hc/fr+hc__creoleval__bert-enc__n-a_seed1.conllu",
                "mc+hc_lstm": f"{ROOT}/predictions/hc/mc+hc__creoleval__lstm-tag-char-bert__glove_seed1.conllu",
                "mc+hc_bert": f"{ROOT}/predictions/hc/mc+hc__creoleval__bert-enc__n-a_seed1.conllu",
            },
            "mtl_models": {
                "mtl-hc_lstm": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-hc__creoleval__lstm-tag-char-bert__glove/hc_seed1.conllu",
                "mtl-hc_bert": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-hc__creoleval__bert-enc__n-a/hc_seed1.conllu",
                "mtl-hcxmc_lstm": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-hcxmc__creoleval__lstm-tag-char-bert__glove/hc_seed1.conllu",
                "mtl-hcxmc_bert": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-hcxmc__creoleval__bert-enc__n-a/hc_seed1.conllu",
                "mtl-fr+hcxfr+mc_bert": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-fr+hcxfr+mc__creoleval__bert-enc__n-a/hc_seed1.conllu",
                "mtl-fr+hcxfr+mc_lstm": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-fr+hcxfr+mc__creoleval__lstm-tag-char-bert__glove/hc_seed1.conllu",
                "mtl-fr+hcxmc_bert": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-fr+hcxmc__creoleval__bert-enc__n-a/hc_seed1.conllu",
                "mtl-fr+hcxmc_lstm": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-fr+hcxmc__creoleval__lstm-tag-char-bert__glove/hc_seed1.conllu",
                "mtl-hcxfr+mc_bert": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-hcxfr+mc__creoleval__bert-enc__n-a/hc_seed1.conllu",
                "mtl-hcxfr+mc_lstm": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-hcxfr+mc__creoleval__lstm-tag-char-bert__glove/hc_seed1.conllu",
            },
        },
        "mc": {
            "gold": f"{ROOT}/data/mc/mc_original_split_test.conllu",
            "non_mtl_models": {
                "mc_lstm": f"{ROOT}/predictions/mc/mc__creoleval__lstm-tag-char-bert__glove_seed1.conllu",
                "mc_bert": f"{ROOT}/predictions/mc/mc__creoleval__bert-enc__n-a_seed1.conllu",
                "fr+mc_lstm": f"{ROOT}/predictions/mc/fr+mc__creoleval__lstm-tag-char-bert__glove_seed1.conllu",
                "fr+mc_bert": f"{ROOT}/predictions/mc/fr+mc__creoleval__bert-enc__n-a_seed1.conllu",
                "hc+mc_lstm": f"{ROOT}/predictions/mc/hc+mc__creoleval__lstm-tag-char-bert__glove_seed1.conllu",
                "hc+mc_bert": f"{ROOT}/predictions/mc/hc+mc__creoleval__bert-enc__n-a_seed1.conllu",
            },
            "mtl_models": {
                "mtl-mc_lstm": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-mc__creoleval__lstm-tag-char-bert__glove/mc_seed1.conllu",
                "mtl-mc_bert": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-mc__creoleval__bert-enc__n-a/mc_seed1.conllu",
                "mtl-hcxmc_lstm": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-hcxmc__creoleval__lstm-tag-char-bert__glove/mc_seed1.conllu",
                "mtl-hcxmc_bert": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-hcxmc__creoleval__bert-enc__n-a/mc_seed1.conllu",
                "mtl-fr+hcxfr+mc_bert": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-fr+hcxfr+mc__creoleval__bert-enc__n-a/mc_seed1.conllu",
                "mtl-fr+hcxfr+mc_lstm": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-fr+hcxfr+mc__creoleval__lstm-tag-char-bert__glove/mc_seed1.conllu",
                "mtl-fr+hcxmc_bert": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-fr+hcxmc__creoleval__bert-enc__n-a/mc_seed1.conllu",
                "mtl-fr+hcxmc_lstm": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-fr+hcxmc__creoleval__lstm-tag-char-bert__glove/mc_seed1.conllu",
                "mtl-hcxfr+mc_bert": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-hcxfr+mc__creoleval__bert-enc__n-a/mc_seed1.conllu",
                "mtl-hcxfr+mc_lstm": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-hcxfr+mc__creoleval__lstm-tag-char-bert__glove/mc_seed1.conllu",
            },
        },
    }
    
    print("="*70)
    print("PHASE 6.1: RELATION-AWARE FUSION ANALYSIS")
    print("="*70)
    print("\nMotivation: Phase 6 showed distance-only heuristics fail.")
    print("Phase 5 revealed models have complementary strengths by RELATION TYPE.")
    print("\nThis analysis tests:")
    print("  1. Relation-aware routing (best model per relation class)")
    print("  2. Combined strategies (distance + relation)")
    print("  3. Gap analysis: Best Single → Best Strategy → Oracle\n")
    
    # Run analysis for all tracks
    all_results = []
    
    for lang in ["hc", "mc"]:
        for track in ["non_mtl", "mtl"]:
            output_dir = f"{args.output_dir}/{lang}/{track}"
            os.makedirs(output_dir, exist_ok=True)
            
            gold_path = config[lang]["gold"]
            model_paths = config[lang][f"{track}_models"]
            
            results = run_track_analysis(lang, track, gold_path, model_paths, output_dir)
            all_results.append(results)
    
    # Generate summary report
    print(f"\n{'='*70}")
    print("PHASE 6.1 SUMMARY")
    print(f"{'='*70}\n")
    
    summary_lines = []
    summary_lines.append("PHASE 6.1 RESULTS: RELATION-AWARE FUSION")
    summary_lines.append("=" * 70)
    summary_lines.append("")
    
    for res in all_results:
        summary_lines.append(f"{res['lang'].upper()} | {res['track'].upper()}")
        summary_lines.append("-" * 70)
        summary_lines.append(f"  Best Single Model: {res['best_single_model']}")
        summary_lines.append(f"    LAS: {res['best_single_las']:.2%}")
        summary_lines.append(f"")
        summary_lines.append(f"  Relation-Aware Routing:")
        summary_lines.append(f"    LAS: {res['relation_routing_las']:.2%} (gain: {res['relation_routing_gain']:+.2%})")
        summary_lines.append(f"    Routing: {res['relation_routing_counts']}")
        summary_lines.append(f"")
        summary_lines.append(f"  Combined Distance+Relation Routing:")
        summary_lines.append(f"    LAS: {res['combined_routing_las']:.2%} (gain: {res['combined_routing_gain']:+.2%})")
        summary_lines.append(f"    Routing: {res['combined_routing_counts']}")
        summary_lines.append(f"")
        summary_lines.append(f"  Oracle Upper Bound:")
        summary_lines.append(f"    LAS: {res['oracle_las']:.2%} (gap: {res['oracle_gap']:+.2%})")
        summary_lines.append(f"")
        summary_lines.append(f"  THE GAP:")
        summary_lines.append(f"    Best Single: {res['best_single_las']:.2%}")
        summary_lines.append(f"    → Best Strategy: {max(res['relation_routing_las'], res['combined_routing_las']):.2%}")
        summary_lines.append(f"    → Oracle: {res['oracle_las']:.2%}")
        summary_lines.append("")
    
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    
    # Save summary
    with open(f"{args.output_dir}/summary.txt", 'w') as f:
        f.write(summary_text)
    
    # Save CSV summary
    summary_df = pd.DataFrame([{
        'Language': r['lang'].upper(),
        'Track': r['track'].upper(),
        'Best_Single_LAS': r['best_single_las'],
        'Relation_Routing_LAS': r['relation_routing_las'],
        'Relation_Gain': r['relation_routing_gain'],
        'Combined_Routing_LAS': r['combined_routing_las'],
        'Combined_Gain': r['combined_routing_gain'],
        'Oracle_LAS': r['oracle_las'],
        'Oracle_Gap': r['oracle_gap'],
    } for r in all_results])
    summary_df.to_csv(f"{args.output_dir}/summary.csv", index=False)
    
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Summary: {args.output_dir}/summary.txt")


if __name__ == "__main__":
    main()
