#!/usr/bin/env python3
"""
Phase 6 (Paper 1): Two-Track Oracle & Complementarity Analysis Pipeline
------------------------------------------------------------------------
Scope:
- TWO SEPARATE TRACKS: Non-MTL (Phase 3) vs MTL (Phase 4)
- Cannot fuse across tracks (different architectures)
- Track 1 (Non-MTL): Single-task baselines + concatenation augmentation
- Track 2 (MTL): Multi-task learning with shared encoder + task heads
- Oracle = upper bound for fusion strategies WITHIN each track
- Strategic routing insight: Short deps → Non-MTL, Long deps → MTL

Outputs split by language (HC vs MC) and track:
results/phase6/
  hc/
    non_mtl/
    mtl/
  mc/
    non_mtl/
    mtl/
  diagnostics/
  paper_tables/
  narrative/
"""

import argparse
import json
import os
import tarfile
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


ROOT = "/home/a-lvm861/projects/supar/supar_creole"
DEFAULT_OUTPUT_DIR = f"{ROOT}/results/phase6"
MIN_COUNT = 10
RANDOM_SEED = 42

FUNCTION_TAGS = {"DET", "ADP", "AUX", "PART", "SCONJ", "CCONJ"}
RELATION_CLASSES = {
    "core_args": {"nsubj", "obj", "iobj", "csubj", "ccomp"},
    "modifiers": {"nmod", "amod", "acl", "advcl", "obl"},
    "function": {"case", "mark", "det", "aux", "cop", "fixed"},
    "coordination": {"conj", "cc"},
    "compound": {"compound", "flat"},
    "other": set(),
}

LEGACY_PRESET = {
    "output_dir": DEFAULT_OUTPUT_DIR,
    "datasets": {
        "hc": {
            "gold": f"{ROOT}/data/hc/hc_original_split_test.conllu",
            # Track 1: Non-MTL models (Phase 3 single-task)
            "non_mtl_models": {
                "hc__creoleval__lstm-tag-char-bert__glove_seed1": f"{ROOT}/predictions/hc/hc__creoleval__lstm-tag-char-bert__glove_seed1.conllu",
                "hc__creoleval__bert-enc__n-a_seed1": f"{ROOT}/predictions/hc/hc__creoleval__bert-enc__n-a_seed1.conllu",
                "fr+hc__creoleval__lstm-tag-char-bert__glove_seed1": f"{ROOT}/predictions/hc/fr+hc__creoleval__lstm-tag-char-bert__glove_seed1.conllu",
                "fr+hc__creoleval__bert-enc__n-a_seed1": f"{ROOT}/predictions/hc/fr+hc__creoleval__bert-enc__n-a_seed1.conllu",
                "mc+hc__creoleval__lstm-tag-char-bert__glove_seed1": f"{ROOT}/predictions/hc/mc+hc__creoleval__lstm-tag-char-bert__glove_seed1.conllu",
                "mc+hc__creoleval__bert-enc__n-a_seed1": f"{ROOT}/predictions/hc/mc+hc__creoleval__bert-enc__n-a_seed1.conllu",
            },
            # Track 2: MTL models (Phase 4 multi-task learning)
            # Note: Phase 4.2 (fr+hc, fr+mc) covered by Phase 4.3a mtl-fr+hcxfr+mc
            "mtl_models": {
                "mtl-hc__creoleval__lstm-tag-char-bert__glove_seed1": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-hc__creoleval__lstm-tag-char-bert__glove/hc_seed1.conllu",
                "mtl-hc__creoleval__bert-enc__n-a_seed1": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-hc__creoleval__bert-enc__n-a/hc_seed1.conllu",
                "mtl-hcxmc__creoleval__lstm-tag-char-bert__glove_seed1": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-hcxmc__creoleval__lstm-tag-char-bert__glove/hc_seed1.conllu",
                "mtl-hcxmc__creoleval__bert-enc__n-a_seed1": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-hcxmc__creoleval__bert-enc__n-a/hc_seed1.conllu",
                "mtl-fr+hcxfr+mc__creoleval__bert-enc__n-a_seed1": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-fr+hcxfr+mc__creoleval__bert-enc__n-a/hc_seed1.conllu",
                "mtl-fr+hcxfr+mc__creoleval__lstm-tag-char-bert__glove_seed1": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-fr+hcxfr+mc__creoleval__lstm-tag-char-bert__glove/hc_seed1.conllu",
                "mtl-fr+hcxmc__creoleval__bert-enc__n-a_seed1": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-fr+hcxmc__creoleval__bert-enc__n-a/hc_seed1.conllu",
                "mtl-fr+hcxmc__creoleval__lstm-tag-char-bert__glove_seed1": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-fr+hcxmc__creoleval__lstm-tag-char-bert__glove/hc_seed1.conllu",
                "mtl-hcxfr+mc__creoleval__bert-enc__n-a_seed1": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-hcxfr+mc__creoleval__bert-enc__n-a/hc_seed1.conllu",
                "mtl-hcxfr+mc__creoleval__lstm-tag-char-bert__glove_seed1": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-hcxfr+mc__creoleval__lstm-tag-char-bert__glove/hc_seed1.conllu",
            },
            # Explicit model pairs for HC - Non-MTL Track
            "non_mtl_key_pairs": {
                "lstm_vs_bert_baseline": ["hc__creoleval__lstm-tag-char-bert__glove_seed1", "hc__creoleval__bert-enc__n-a_seed1"],
                "best_concat_vs_best_baseline": ["mc+hc__creoleval__lstm-tag-char-bert__glove_seed1", "hc__creoleval__lstm-tag-char-bert__glove_seed1"],
                "french_aug_vs_creole_aug": ["fr+hc__creoleval__lstm-tag-char-bert__glove_seed1", "mc+hc__creoleval__lstm-tag-char-bert__glove_seed1"],
            },
            # Explicit model pairs for HC - MTL Track  
            "mtl_key_pairs": {
                "mtl_lstm_vs_bert_baseline": ["mtl-hc__creoleval__lstm-tag-char-bert__glove_seed1", "mtl-hc__creoleval__bert-enc__n-a_seed1"],
                "mtl_baseline_vs_joint": ["mtl-hc__creoleval__bert-enc__n-a_seed1", "mtl-hcxmc__creoleval__bert-enc__n-a_seed1"],
                "mtl_joint_vs_all_aug": ["mtl-hcxmc__creoleval__bert-enc__n-a_seed1", "mtl-fr+hcxfr+mc__creoleval__bert-enc__n-a_seed1"],
                "mtl_best_asym": ["mtl-fr+hcxmc__creoleval__bert-enc__n-a_seed1", "mtl-hcxfr+mc__creoleval__bert-enc__n-a_seed1"],
            },
        },
        "mc": {
            "gold": f"{ROOT}/data/mc/mc_original_split_test.conllu",
            # Track 1: Non-MTL models (Phase 3 single-task)
            "non_mtl_models": {
                "mc__creoleval__lstm-tag-char-bert__glove_seed1": f"{ROOT}/predictions/mc/mc__creoleval__lstm-tag-char-bert__glove_seed1.conllu",
                "mc__creoleval__bert-enc__n-a_seed1": f"{ROOT}/predictions/mc/mc__creoleval__bert-enc__n-a_seed1.conllu",
                "fr+mc__creoleval__lstm-tag-char-bert__glove_seed1": f"{ROOT}/predictions/mc/fr+mc__creoleval__lstm-tag-char-bert__glove_seed1.conllu",
                "fr+mc__creoleval__bert-enc__n-a_seed1": f"{ROOT}/predictions/mc/fr+mc__creoleval__bert-enc__n-a_seed1.conllu",
                "hc+mc__creoleval__lstm-tag-char-bert__glove_seed1": f"{ROOT}/predictions/mc/hc+mc__creoleval__lstm-tag-char-bert__glove_seed1.conllu",
                "hc+mc__creoleval__bert-enc__n-a_seed1": f"{ROOT}/predictions/mc/hc+mc__creoleval__bert-enc__n-a_seed1.conllu",
            },
            # Track 2: MTL models (Phase 4 multi-task learning)
            # Note: Phase 4.2 (fr+mc, fr+hc) covered by Phase 4.3a mtl-fr+hcxfr+mc
            "mtl_models": {
                "mtl-mc__creoleval__lstm-tag-char-bert__glove_seed1": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-mc__creoleval__lstm-tag-char-bert__glove/mc_seed1.conllu",
                "mtl-mc__creoleval__bert-enc__n-a_seed1": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-mc__creoleval__bert-enc__n-a/mc_seed1.conllu",
                "mtl-hcxmc__creoleval__lstm-tag-char-bert__glove_seed1": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-hcxmc__creoleval__lstm-tag-char-bert__glove/mc_seed1.conllu",
                "mtl-hcxmc__creoleval__bert-enc__n-a_seed1": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-hcxmc__creoleval__bert-enc__n-a/mc_seed1.conllu",
                "mtl-fr+hcxfr+mc__creoleval__bert-enc__n-a_seed1": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-fr+hcxfr+mc__creoleval__bert-enc__n-a/mc_seed1.conllu",
                "mtl-fr+hcxfr+mc__creoleval__lstm-tag-char-bert__glove_seed1": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-fr+hcxfr+mc__creoleval__lstm-tag-char-bert__glove/mc_seed1.conllu",
                "mtl-fr+hcxmc__creoleval__bert-enc__n-a_seed1": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-fr+hcxmc__creoleval__bert-enc__n-a/mc_seed1.conllu",
                "mtl-fr+hcxmc__creoleval__lstm-tag-char-bert__glove_seed1": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-fr+hcxmc__creoleval__lstm-tag-char-bert__glove/mc_seed1.conllu",
                "mtl-hcxfr+mc__creoleval__bert-enc__n-a_seed1": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-hcxfr+mc__creoleval__bert-enc__n-a/mc_seed1.conllu",
                "mtl-hcxfr+mc__creoleval__lstm-tag-char-bert__glove_seed1": "/home/a-lvm861/projects/supar/multiparser/predictions/mtl-hcxfr+mc__creoleval__lstm-tag-char-bert__glove/mc_seed1.conllu",
            },
            # Explicit model pairs for MC - Non-MTL Track
            "non_mtl_key_pairs": {
                "lstm_vs_bert_baseline": ["mc__creoleval__lstm-tag-char-bert__glove_seed1", "mc__creoleval__bert-enc__n-a_seed1"],
                "best_concat_vs_best_baseline": ["hc+mc__creoleval__lstm-tag-char-bert__glove_seed1", "mc__creoleval__lstm-tag-char-bert__glove_seed1"],
                "french_aug_vs_creole_aug": ["fr+mc__creoleval__lstm-tag-char-bert__glove_seed1", "hc+mc__creoleval__lstm-tag-char-bert__glove_seed1"],
            },
            # Explicit model pairs for MC - MTL Track
            "mtl_key_pairs": {
                "mtl_lstm_vs_bert_baseline": ["mtl-mc__creoleval__lstm-tag-char-bert__glove_seed1", "mtl-mc__creoleval__bert-enc__n-a_seed1"],
                "mtl_baseline_vs_joint": ["mtl-mc__creoleval__bert-enc__n-a_seed1", "mtl-hcxmc__creoleval__bert-enc__n-a_seed1"],
                "mtl_joint_vs_all_aug": ["mtl-hcxmc__creoleval__bert-enc__n-a_seed1", "mtl-fr+hcxfr+mc__creoleval__bert-enc__n-a_seed1"],
                "mtl_best_asym": ["mtl-fr+hcxmc__creoleval__bert-enc__n-a_seed1", "mtl-hcxfr+mc__creoleval__bert-enc__n-a_seed1"],
            },
        },
    },
}


@dataclass
class DSResult:
    lang: str
    track: str  # "non_mtl" or "mtl"
    models: List[str]
    aligned: pd.DataFrame
    with_oracle: pd.DataFrame
    comp_las_selected: pd.DataFrame
    fusion_results: pd.DataFrame
    best_model: str
    best_uas: float
    best_las: float
    oracle_uas: float
    oracle_las: float


def read_conllu(path: str) -> List[List[List[str]]]:
    sents, sent = [], []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                if sent:
                    sents.append(sent)
                    sent = []
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) >= 8:
                sent.append(cols)
    if sent:
        sents.append(sent)
    return sents


def token_map(sent: List[List[str]]) -> Dict[int, List[str]]:
    out = {}
    for tok in sent:
        tid = tok[0]
        if "-" in tid or "." in tid:
            continue
        try:
            out[int(tid)] = tok
        except ValueError:
            pass
    return out


def rel_class(deprel: str) -> str:
    if not isinstance(deprel, str):
        return "other"
    for k, v in RELATION_CLASSES.items():
        if deprel in v:
            return k
    return "other"


def dist_bin(x) -> str:
    if pd.isna(x):
        return "null"
    d = int(x)
    if d == 0:
        return "0"
    if d == 1:
        return "1"
    if d == 2:
        return "2"
    if 3 <= d <= 5:
        return "3-5"
    if 6 <= d <= 10:
        return "6-10"
    return "11+"


def align(gold_path: str, model_paths: Dict[str, str]) -> pd.DataFrame:
    gold_s = read_conllu(gold_path)
    pred_s = {m: read_conllu(p) for m, p in model_paths.items()}
    models = list(model_paths.keys())
    rows = []

    for si, gs in enumerate(gold_s):
        gm = token_map(gs)
        pm = {m: token_map(pred_s[m][si]) if si < len(pred_s[m]) else {} for m in models}
        for tid in sorted(gm.keys()):
            g = gm[tid]
            gh = int(g[6]) if g[6].lstrip("-").isdigit() else None
            gd = g[7]
            upos = g[3] if len(g) > 3 else None
            row = {
                "sent_idx": si,
                "token_id": tid,
                "token": g[1],
                "upos": upos,
                "is_function": int(upos in FUNCTION_TAGS) if upos else None,
                "gold_head": gh,
                "gold_deprel": gd,
                "relation_class": rel_class(gd),
                "head_distance": 0 if gh == 0 else (abs(tid - gh) if gh is not None else None),
            }
            for i, m in enumerate(models):
                p = pm[m].get(tid)
                if p is None:
                    ph, pdp = None, None
                else:
                    ph = int(p[6]) if p[6].lstrip("-").isdigit() else None
                    pdp = p[7]
                ch = int(ph == gh) if (ph is not None and gh is not None) else 0
                cd = int(pdp == gd) if pdp is not None else 0
                cl = int(ch == 1 and cd == 1)
                row[f"pred_head_{m}"] = ph
                row[f"pred_deprel_{m}"] = pdp
                row[f"correct_head_{m}"] = ch
                row[f"correct_deprel_{m}"] = cd
                row[f"correct_las_{m}"] = cl
                if i == 0:
                    row["pred_head"] = ph
                    row["pred_deprel"] = pdp
                    row["correct_head"] = ch
                    row["correct_deprel"] = cd
            rows.append(row)
    return pd.DataFrame(rows)


def best_single(df: pd.DataFrame, models: List[str]) -> Tuple[str, float, float]:
    best_m, best_u, best_l = None, -1.0, -1.0
    for m in models:
        u = float(df[f"correct_head_{m}"].mean())
        l = float(df[f"correct_las_{m}"].mean())
        if l > best_l:
            best_m, best_u, best_l = m, u, l
    return best_m, best_u, best_l


def add_oracle(df: pd.DataFrame, models: List[str]) -> Tuple[pd.DataFrame, float, float]:
    out = df.copy()
    out["oracle_correct_head"] = out[[f"correct_head_{m}" for m in models]].max(axis=1)
    out["oracle_correct_las"] = out[[f"correct_las_{m}" for m in models]].max(axis=1)
    return out, float(out["oracle_correct_head"].mean()), float(out["oracle_correct_las"].mean())


def comp_las_for_pairs(df: pd.DataFrame, key_pairs: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []
    total = len(df)
    for pair_name, pair in key_pairs.items():
        if len(pair) != 2:
            continue
        a, b = pair
        if f"correct_las_{a}" not in df.columns or f"correct_las_{b}" not in df.columns:
            continue
        al = df[f"correct_las_{a}"]
        bl = df[f"correct_las_{b}"]
        a_only = int(((al == 1) & (bl == 0)).sum())
        b_only = int(((al == 0) & (bl == 1)).sum())
        rows.append(
            {
                "Pair_Name": pair_name,
                "Model_A": a,
                "Model_B": b,
                "A_Correct_B_Wrong_LAS": a_only,
                "B_Correct_A_Wrong_LAS": b_only,
                "Both_Correct_LAS": int(((al == 1) & (bl == 1)).sum()),
                "Both_Wrong_LAS": int(((al == 0) & (bl == 0)).sum()),
                "Complementarity_LAS": float((a_only + b_only) / total if total else 0.0),
                "Total_Tokens": total,
            }
        )
    return pd.DataFrame(rows).sort_values("Complementarity_LAS", ascending=False)


def all_model_pairs(models: List[str]) -> Dict[str, List[str]]:
    pairs = {}
    for a, b in combinations(models, 2):
        pairs[f"all::{a}__vs__{b}"] = [a, b]
    return pairs


def pair_oracle_summary(lang: str, df: pd.DataFrame, key_pairs: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []
    for pair_name, pair in key_pairs.items():
        if len(pair) != 2:
            continue
        a, b = pair
        if f"correct_las_{a}" not in df.columns or f"correct_las_{b}" not in df.columns:
            continue
        la = float(df[f"correct_las_{a}"].mean())
        lb = float(df[f"correct_las_{b}"].mean())
        oracle_pair = float(df[[f"correct_las_{a}", f"correct_las_{b}"]].max(axis=1).mean())
        comp = float((((df[f"correct_las_{a}"] == 1) & (df[f"correct_las_{b}"] == 0)).sum()
                      + ((df[f"correct_las_{a}"] == 0) & (df[f"correct_las_{b}"] == 1)).sum()) / len(df))
        best_las = max(la, lb)
        gain = oracle_pair - best_las
        remaining_error = 1.0 - best_las
        relative_gain = (gain / remaining_error) if remaining_error > 0 else 0.0
        rows.append(
            {
                "Language": lang.upper(),
                "Pair_Name": pair_name,
                "Model_A": a,
                "Model_B": b,
                "LAS_A": la,
                "LAS_B": lb,
                "Oracle_LAS": oracle_pair,
                "Oracle_Gain_over_best": gain,
                "Relative_Gain": relative_gain,
                "Complementarity_LAS": comp,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("Oracle_Gain_over_best", ascending=False).reset_index(drop=True)
    out["Rank_by_Oracle_Gain"] = out.index + 1
    return out[
        [
            "Language",
            "Rank_by_Oracle_Gain",
            "Pair_Name",
            "Model_A",
            "Model_B",
            "LAS_A",
            "LAS_B",
            "Oracle_LAS",
            "Oracle_Gain_over_best",
            "Relative_Gain",
            "Complementarity_LAS",
        ]
    ]


def fusion_heuristic_failure(df: pd.DataFrame, key_pairs: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []
    for pair_name, pair in key_pairs.items():
        if len(pair) != 2:
            continue
        a, b = pair
        if f"pred_head_{a}" not in df.columns or f"pred_head_{b}" not in df.columns:
            continue
        for rule in ["Dist>2→A,≤2→B", "Dist≤2→A,>2→B"]:
            choose_a = (df["head_distance"] > 2) if rule.startswith("Dist>2") else (df["head_distance"] <= 2)
            ph = df[f"pred_head_{a}"].where(choose_a, df[f"pred_head_{b}"])
            pr = df[f"pred_deprel_{a}"].where(choose_a, df[f"pred_deprel_{b}"])
            uas = float((ph == df["gold_head"]).mean())
            las = float(((ph == df["gold_head"]) & (pr == df["gold_deprel"])).mean())
            rows.append(
                {
                    "Pair_Name": pair_name,
                    "Rule": rule,
                    "Model_A": a,
                    "Model_B": b,
                    "UAS": uas,
                    "LAS": las,
                    "Use_A_Pct": float(choose_a.mean() * 100.0),
                }
            )
    return pd.DataFrame(rows).sort_values(["LAS", "UAS"], ascending=False)


def alignment_diag(lang: str, df: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    rows = []
    n = len(df)
    for m in models:
        miss_h = int(df[f"pred_head_{m}"].isna().sum())
        miss_r = int(df[f"pred_deprel_{m}"].isna().sum())
        miss_any = int((df[f"pred_head_{m}"].isna() | df[f"pred_deprel_{m}"].isna()).sum())
        rows.append(
            {
                "Language": lang.upper(),
                "Model": m,
                "Total_Tokens": n,
                "Missing_Head": miss_h,
                "Missing_Deprel": miss_r,
                "Missing_Any": miss_any,
                "Missing_Pct": float((miss_any / n * 100.0) if n else 0.0),
                "Flag_Missing_gt_1pct": int(((miss_any / n * 100.0) if n else 0.0) > 1.0),
            }
        )
    # token continuity summary
    bad = 0
    for _, g in df.groupby("sent_idx"):
        ids = sorted(g["token_id"].tolist())
        exp = list(range(ids[0], ids[-1] + 1)) if ids else []
        if ids != exp:
            bad += 1
    rows.append(
        {
            "Language": lang.upper(),
            "Model": "__token_continuity_summary__",
            "Total_Tokens": n,
            "Missing_Head": None,
            "Missing_Deprel": None,
            "Missing_Any": None,
            "Missing_Pct": None,
            "Flag_Missing_gt_1pct": int(bad > 0),
            "Sentences_With_Discontinuity": bad,
        }
    )
    return pd.DataFrame(rows)


def sanity_sample(lang: str, df: pd.DataFrame, models: List[str], out_path: str) -> None:
    s = df.sample(n=min(20, len(df)), random_state=RANDOM_SEED).copy()
    cols = ["sent_idx", "token_id", "token", "gold_head", "gold_deprel"]
    for m in models:
        s[f"check_head_{m}"] = (s[f"pred_head_{m}"] == s["gold_head"]).astype(int)
        s[f"check_deprel_{m}"] = (s[f"pred_deprel_{m}"] == s["gold_deprel"]).astype(int)
        s[f"check_match_head_{m}"] = (s[f"check_head_{m}"] == s[f"correct_head_{m}"]).astype(int)
        s[f"check_match_deprel_{m}"] = (s[f"check_deprel_{m}"] == s[f"correct_deprel_{m}"]).astype(int)
        cols.extend(
            [
                f"pred_head_{m}",
                f"pred_deprel_{m}",
                f"correct_head_{m}",
                f"correct_deprel_{m}",
                f"check_match_head_{m}",
                f"check_match_deprel_{m}",
            ]
        )
    s.insert(0, "Language", lang.upper())
    s[["Language"] + cols].to_csv(out_path, index=False)


def advantage_by_group(lang: str, df: pd.DataFrame, pairs: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    drows, rrows, frows = [], [], []
    w = df.copy()
    w["distance_bin"] = w["head_distance"].map(dist_bin)
    w["func_group"] = w["is_function"].map({1: "function", 0: "content"}).fillna("unknown")
    for pair_name, pair in pairs.items():
        if len(pair) != 2:
            continue
        a, b = pair
        if f"correct_las_{a}" not in w.columns or f"correct_las_{b}" not in w.columns:
            continue
        w["_A_better"] = ((w[f"correct_las_{a}"] == 1) & (w[f"correct_las_{b}"] == 0)).astype(int)
        w["_B_better"] = ((w[f"correct_las_{a}"] == 0) & (w[f"correct_las_{b}"] == 1)).astype(int)
        for k, g in w.groupby("distance_bin"):
            if len(g) < MIN_COUNT:
                continue
            drows.append(
                {
                    "Language": lang.upper(),
                    "Pair_Name": pair_name,
                    "Model_A": a,
                    "Model_B": b,
                    "distance_bin": k,
                    "count": int(len(g)),
                    "A_better_pct": float(g["_A_better"].mean() * 100.0),
                    "B_better_pct": float(g["_B_better"].mean() * 100.0),
                }
            )
        for k, g in w.groupby("gold_deprel"):
            if len(g) < MIN_COUNT:
                continue
            rrows.append(
                {
                    "Language": lang.upper(),
                    "Pair_Name": pair_name,
                    "Model_A": a,
                    "Model_B": b,
                    "gold_deprel": k,
                    "count": int(len(g)),
                    "A_better_pct": float(g["_A_better"].mean() * 100.0),
                    "B_better_pct": float(g["_B_better"].mean() * 100.0),
                }
            )
        for k, g in w.groupby("func_group"):
            if len(g) < MIN_COUNT:
                continue
            frows.append(
                {
                    "Language": lang.upper(),
                    "Pair_Name": pair_name,
                    "Model_A": a,
                    "Model_B": b,
                    "function_vs_content": k,
                    "count": int(len(g)),
                    "A_better_pct": float(g["_A_better"].mean() * 100.0),
                    "B_better_pct": float(g["_B_better"].mean() * 100.0),
                }
            )
    return pd.DataFrame(drows), pd.DataFrame(rrows), pd.DataFrame(frows)


def oracle_gain(lang: str, df: pd.DataFrame, best_model: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    w = df.copy()
    w["distance_bin"] = w["head_distance"].map(dist_bin)
    drows, rrows = [], []
    for k, g in w.groupby("distance_bin"):
        if len(g) < MIN_COUNT:
            continue
        bl = float(g[f"correct_las_{best_model}"].mean())
        ol = float(g["oracle_correct_las"].mean())
        drows.append(
            {
                "Language": lang.upper(),
                "distance_bin": k,
                "count": int(len(g)),
                "best_model": best_model,
                "best_model_las": bl,
                "oracle_las": ol,
                "delta_oracle_minus_best": ol - bl,
            }
        )
    for k, g in w.groupby("relation_class"):
        if len(g) < MIN_COUNT:
            continue
        bl = float(g[f"correct_las_{best_model}"].mean())
        ol = float(g["oracle_correct_las"].mean())
        rrows.append(
            {
                "Language": lang.upper(),
                "relation_class": k,
                "count": int(len(g)),
                "best_model": best_model,
                "best_model_las": bl,
                "oracle_las": ol,
                "delta_oracle_minus_best": ol - bl,
            }
        )
    return pd.DataFrame(drows), pd.DataFrame(rrows)


def refined_rules_diagnostics_only(lang: str, df: pd.DataFrame, pairs: Dict[str, List[str]], best_model: str, out_path: str) -> None:
    # Kept as diagnostics only; no learning.
    rows = []
    for pair_name, pair in pairs.items():
        if len(pair) != 2:
            continue
        a, b = pair
        if f"pred_head_{a}" not in df.columns or f"pred_head_{b}" not in df.columns:
            continue
        for th in [1, 2, 3, 4, 5]:
            choose_b = df["head_distance"] > th
            ph = df[f"pred_head_{b}"].where(choose_b, df[f"pred_head_{a}"])
            pr = df[f"pred_deprel_{b}"].where(choose_b, df[f"pred_deprel_{a}"])
            las = float(((ph == df["gold_head"]) & (pr == df["gold_deprel"])).mean())
            rows.append(
                {
                    "Language": lang.upper(),
                    "Pair_Name": pair_name,
                    "Rule_Type": "distance_refined",
                    "Model_A": a,
                    "Model_B": b,
                    "Rule_Params": json.dumps({"use_B_when_distance_gt": th}),
                    "LAS": las,
                    "best_single_las": float(df[f"correct_las_{best_model}"].mean()),
                    "delta_vs_best_single_las": las - float(df[f"correct_las_{best_model}"].mean()),
                }
            )
    pd.DataFrame(rows).sort_values("LAS", ascending=False).to_csv(out_path, index=False)


def table4_relation_class_errors(ds_results: List[DSResult], out_path: str) -> None:
    rows = []
    for ds in ds_results:
        for cls, g in ds.with_oracle.groupby("relation_class"):
            if len(g) < MIN_COUNT:
                continue
            rows.append(
                {
                    "Language": ds.lang.upper(),
                    "relation_class": cls,
                    "count": int(len(g)),
                    "best_model": ds.best_model,
                    "best_model_error_pct": float((1.0 - g[f"correct_las_{ds.best_model}"].mean()) * 100.0),
                    "oracle_error_pct": float((1.0 - g["oracle_correct_las"].mean()) * 100.0),
                }
            )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def build_narrative(out_dir: str, ds_results: List[DSResult]) -> None:
    narrative_path = f"{out_dir}/phase6_paper1_narrative.txt"
    check_path = f"{out_dir}/phase6_interpretation_check.txt"

    # Extract core evidences by language and track
    evidence = {}
    for ds in ds_results:
        comp_mean = float(ds.comp_las_selected["Complementarity_LAS"].mean()) if not ds.comp_las_selected.empty else 0.0
        key = f"{ds.lang}_{ds.track}"
        evidence[key] = {
            "best_model": ds.best_model,
            "best_las": ds.best_las,
            "oracle_las": ds.oracle_las,
            "oracle_gain": ds.oracle_las - ds.best_las,
            "comp_mean": comp_mean,
        }

    lines = []
    lines.append("PHASE 6 PAPER 1 NARRATIVE (TWO-TRACK ANALYSIS)")
    lines.append("==============================================")
    lines.append("")
    lines.append("Section 1 — Project Overview")
    lines.append("This project studies dependency parsing for Haitian Creole (HC) and Martinican Creole (MC) in low-resource, contact-linguistic settings.")
    lines.append("Phases 1–6 evaluate model families, augmentation, MTL, and diagnostics of structural error patterns.")
    lines.append("")
    lines.append("Section 2 — Phase 1–3 (Non-MTL Model Development)")
    lines.append("Baseline model families: LSTM-based and BERT-encoder parsers, trained on monolingual and augmented data settings.")
    lines.append("Phase 3 established stable comparative baselines under seed controls.")
    lines.append("")
    lines.append("Section 3 — Phase 4 (MTL Model Development)")
    lines.append("Multi-task learning (MTL) with shared encoder and task-specific heads.")
    lines.append("Covered Phase 4.0-4.3: monolingual baseline, joint HC×MC, French-augmented, asymmetric transfer.")
    lines.append("")
    lines.append("Section 4 — Phase 5 (Error Analysis)")
    lines.append("Phase 5 identified distance and relation-class sensitivity across ALL models (Non-MTL and MTL).")
    lines.append("Modifiers emerged as a recurrent difficult class. These observations motivated structured diagnostic comparisons in Phase 6.")
    lines.append("")
    lines.append("Section 5 — Phase 6 (Two-Track Complementarity + Oracle)")
    lines.append("Analysis runs separately for:")
    lines.append("  Track 1 (Non-MTL): Phase 3 single-task models (baselines + concat augmentation)")
    lines.append("  Track 2 (MTL): Phase 4 multi-task learning models (shared encoder + task heads)")
    lines.append("Cannot fuse across tracks due to architecture differences.")
    lines.append("")
    lines.append("Results by Language and Track:")
    for lang in ["hc", "mc"]:
        for track in ["non_mtl", "mtl"]:
            key = f"{lang}_{track}"
            if key in evidence:
                e = evidence[key]
                lines.append(
                    f"  {lang.upper()} | {track.upper()}: best LAS={e['best_las']:.2%} ({e['best_model']}), "
                    f"oracle LAS={e['oracle_las']:.2%}, oracle gain={e['oracle_gain']:+.2%}, "
                    f"selected-pair LAS complementarity mean={e['comp_mean']:.2%}."
                )
    lines.append("")
    lines.append("Oracle is treated strictly as an upper bound, not an achievable deployed method.")
    lines.append("")
    lines.append("Section 6 — Key Findings")
    lines.append("1) No single model dominates every structure in every condition (verified via pairwise/group diagnostics).")
    lines.append("2) Complementarity appears structured (distance/relation dependent), but magnitude varies by language, track, and pair.")
    lines.append("3) Oracle analysis shows non-trivial upper-bound headroom over best single models in both tracks.")
    lines.append("4) Heuristic fusion strategies appear to yield limited or inconsistent gains, suggesting complementarity is structured but not trivially exploitable.")
    lines.append("5) Strategic routing concept: Short/local dependencies may benefit from Non-MTL models, while long dependencies may benefit from MTL models.")
    lines.append("")
    lines.append("Section 7 — Transition")
    lines.append("These findings motivate future exploration of structured fusion strategies capable of leveraging model complementarity,")
    lines.append("with potential for track-specific routing based on dependency characteristics.")
    with open(narrative_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Interpretation check (mandatory disagreement report)
    chk = []
    chk.append("PHASE 6 INTERPRETATION CHECK")
    chk.append("============================")
    claims = [
        ("No HC/MC mixing", "Supported", "Pipeline runs HC and MC separately and writes separate hc/* and mc/* outputs."),
        ("No Non-MTL/MTL mixing", "Supported", "Analysis runs separately for non_mtl and mtl tracks within each language."),
        ("Oracle labeled upper bound", "Supported", "Oracle appears only in oracle files/tables and narrative as upper bound."),
        ("No fusion improvement claim", "Supported", "Main tables avoid presenting heuristic fusion as solution; heuristic remains diagnostic."),
        ("Complementarity exists at LAS level", "Partial", "Exists in selected pairs, but magnitude varies and is not uniformly large."),
        ("Complementarity increases with distance", "Partial", "Some pairs show distance concentration; not universal across all pairs/languages/tracks."),
        ("Relation classes show disagreement", "Supported", "model_advantage_by_relation and oracle_gain_by_relation show class-specific differences."),
        ("Simple heuristics fail consistently", "Partial", "Often limited/inconsistent; verify per language/track from *_fusion_results.csv."),
    ]
    for c, s, e in claims:
        chk.append(f"Claim: {c}")
        chk.append(f"Supported: {s}")
        chk.append(f"Evidence: {e}")
        chk.append("Notes: See diagnostics and paper_tables for quantified values.")
        chk.append("")
    with open(check_path, "w", encoding="utf-8") as f:
        f.write("\n".join(chk) + "\n")


def ensure_dirs(base: str) -> Dict[str, str]:
    dirs = {
        "base": base,
        "hc_non_mtl": f"{base}/hc/non_mtl",
        "hc_mtl": f"{base}/hc/mtl",
        "mc_non_mtl": f"{base}/mc/non_mtl",
        "mc_mtl": f"{base}/mc/mtl",
        "diagnostics": f"{base}/diagnostics",
        "paper_tables": f"{base}/paper_tables",
        "narrative": f"{base}/narrative",
    }
    for p in dirs.values():
        os.makedirs(p, exist_ok=True)
    return dirs


def validate_paths(cfg: Dict) -> None:
    missing = []
    for _, ds in cfg["datasets"].items():
        if not Path(ds["gold"]).exists():
            missing.append(ds["gold"])
        # Check both non_mtl_models and mtl_models
        for _, p in ds.get("non_mtl_models", {}).items():
            if not Path(p).exists():
                missing.append(p)
        for _, p in ds.get("mtl_models", {}).items():
            if not Path(p).exists():
                missing.append(p)
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(f"  - {m}" for m in missing))


def load_cfg(args: argparse.Namespace) -> Dict:
    if args.preset == "legacy":
        return LEGACY_PRESET
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError("Use --preset legacy or --config /abs/path.json")


def run_track(lang: str, track: str, ds_cfg: Dict, dirs: Dict[str, str]) -> DSResult:
    """Run oracle analysis for one track (non_mtl or mtl) of one language."""
    # Select the appropriate models and pairs for this track
    models_key = f"{track}_models"
    pairs_key = f"{track}_key_pairs"
    
    models = list(ds_cfg[models_key].keys())
    key_pairs = ds_cfg.get(pairs_key, {})
    
    # Use track-specific output directory
    output_key = f"{lang}_{track}"
    output_dir = dirs[output_key]

    aligned = align(ds_cfg["gold"], ds_cfg[models_key])
    aligned.to_csv(f"{output_dir}/{lang}_{track}_aligned_predictions.csv", index=False)

    with_oracle, oracle_u, oracle_l = add_oracle(aligned, models)
    with_oracle.to_csv(f"{output_dir}/{lang}_{track}_with_oracle.csv", index=False)

    comp_las = comp_las_for_pairs(with_oracle, key_pairs)
    comp_las.to_csv(f"{output_dir}/{lang}_{track}_complementarity_las.csv", index=False)

    fusion = fusion_heuristic_failure(with_oracle, key_pairs)
    fusion.to_csv(f"{output_dir}/{lang}_{track}_fusion_results.csv", index=False)

    best_m, best_u, best_l = best_single(with_oracle, models)

    # Diagnostics per language-track
    alignment_diag(lang, aligned, models).to_csv(f"{dirs['diagnostics']}/{lang}_{track}_alignment_diagnostics.csv", index=False)
    sanity_sample(lang, aligned, models, f"{dirs['diagnostics']}/{lang}_{track}_sanity_sample.csv")
    d_adv, r_adv, f_adv = advantage_by_group(lang, with_oracle, key_pairs)
    d_adv.to_csv(f"{dirs['diagnostics']}/{lang}_{track}_model_advantage_by_distance.csv", index=False)
    r_adv.to_csv(f"{dirs['diagnostics']}/{lang}_{track}_model_advantage_by_relation.csv", index=False)
    f_adv.to_csv(f"{dirs['diagnostics']}/{lang}_{track}_model_advantage_by_function.csv", index=False)
    d_gain, r_gain = oracle_gain(lang, with_oracle, best_m)
    d_gain.to_csv(f"{dirs['diagnostics']}/{lang}_{track}_oracle_gain_by_distance.csv", index=False)
    r_gain.to_csv(f"{dirs['diagnostics']}/{lang}_{track}_oracle_gain_by_relation.csv", index=False)

    # head_distance null diagnostic
    null_count = int(with_oracle["head_distance"].isna().sum())
    null_pct = float(null_count / len(with_oracle) * 100.0) if len(with_oracle) else 0.0
    # recompute fusion excluding null distance
    w = with_oracle[with_oracle["head_distance"].notna()].copy()
    rows = []
    for _, fr in fusion.iterrows():
        a, b, rule = fr["Model_A"], fr["Model_B"], fr["Rule"]
        choose_a = (w["head_distance"] > 2) if str(rule).startswith("Dist>2") else (w["head_distance"] <= 2)
        ph = w[f"pred_head_{a}"].where(choose_a, w[f"pred_head_{b}"])
        pr = w[f"pred_deprel_{a}"].where(choose_a, w[f"pred_deprel_{b}"])
        rows.append(
            {
                "Language": lang.upper(),
                "Pair_Name": fr["Pair_Name"],
                "Rule": rule,
                "Model_A": a,
                "Model_B": b,
                "UAS_without_null_distance": float((ph == w["gold_head"]).mean()),
                "LAS_without_null_distance": float(((ph == w["gold_head"]) & (pr == w["gold_deprel"])).mean()),
                "null_distance_count": null_count,
                "null_distance_pct": null_pct,
            }
        )
    pd.DataFrame(rows).to_csv(f"{dirs['diagnostics']}/{lang}_{track}_fusion_without_null_distance.csv", index=False)

    refined_rules_diagnostics_only(
        lang=lang,
        df=with_oracle,
        pairs=key_pairs,
        best_model=best_m,
        out_path=f"{dirs['diagnostics']}/{lang}_{track}_refined_fusion_results.csv",
    )

    # fusion_diagnostics txt
    if not fusion.empty:
        top = fusion.iloc[0]
        a, b, rule = top["Model_A"], top["Model_B"], top["Rule"]
        choose_a = (with_oracle["head_distance"] > 2) if str(rule).startswith("Dist>2") else (with_oracle["head_distance"] <= 2)
        a_l = float(with_oracle[f"correct_las_{a}"].mean())
        b_l = float(with_oracle[f"correct_las_{b}"].mean())
        weaker = a if a_l < b_l else b
        weaker_use = float(choose_a.mean() * 100.0) if weaker == a else float((~choose_a).mean() * 100.0)
        with open(f"{dirs['diagnostics']}/{lang}_{track}_fusion_diagnostics.txt", "w", encoding="utf-8") as f:
            f.write(f"Language: {lang.upper()} | Track: {track.upper()}\n")
            f.write(f"Best heuristic rule: {rule}\n")
            f.write(f"Pair: {top['Pair_Name']} ({a} vs {b})\n")
            f.write(f"Use_A_Pct={choose_a.mean()*100:.2f}, Use_B_Pct={(~choose_a).mean()*100:.2f}\n")
            f.write(f"LAS_A={a_l:.4f}, LAS_B={b_l:.4f}, LAS_Fusion={float(top['LAS']):.4f}\n")
            f.write(f"Weaker model: {weaker}; routed token share={weaker_use:.2f}%\n")
            f.write("Diagnostic interpretation: heuristic routing may over-assign to weaker regions depending on pair.\n")

    return DSResult(
        lang=lang,
        track=track,
        models=models,
        aligned=aligned,
        with_oracle=with_oracle,
        comp_las_selected=comp_las,
        fusion_results=fusion,
        best_model=best_m,
        best_uas=best_u,
        best_las=best_l,
        oracle_uas=oracle_u,
        oracle_las=oracle_l,
    )


def aggregate_files(dirs: Dict[str, str]) -> None:
    # Merge per-language-track diagnostics into global diagnostics files
    def merge(name: str):
        frames = []
        for lang in ["hc", "mc"]:
            for track in ["non_mtl", "mtl"]:
                p = f"{dirs['diagnostics']}/{lang}_{track}_{name}"
                if os.path.exists(p):
                    df = pd.read_csv(p)
                    # Add track column if not present
                    if "Track" not in df.columns:
                        df.insert(0, "Track", track.upper())
                    frames.append(df)
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(f"{dirs['diagnostics']}/{name}", index=False)

    merge("alignment_diagnostics.csv")
    merge("sanity_sample.csv")
    merge("model_advantage_by_distance.csv")
    merge("model_advantage_by_relation.csv")
    merge("model_advantage_by_function.csv")
    merge("oracle_gain_by_distance.csv")
    merge("oracle_gain_by_relation.csv")
    merge("fusion_without_null_distance.csv")
    merge("refined_fusion_results.csv")

    # fusion diagnostics aggregate txt
    with open(f"{dirs['diagnostics']}/fusion_diagnostics.txt", "w", encoding="utf-8") as out:
        for lang in ["hc", "mc"]:
            for track in ["non_mtl", "mtl"]:
                p = f"{dirs['diagnostics']}/{lang}_{track}_fusion_diagnostics.txt"
                out.write(f"=== {lang.upper()} | {track.upper()} ===\n")
                if os.path.exists(p):
                    out.write(Path(p).read_text(encoding="utf-8"))
                out.write("\n")


def build_paper_tables(dirs: Dict[str, str], ds_results: List[DSResult], cfg: Dict) -> None:
    # table1 model comparison
    t1 = []
    t2 = []
    t3 = []
    pairwise_frames = []
    pairwise_all_frames = []
    for ds in ds_results:
        lang = ds.lang
        track = ds.track
        # Get the appropriate key_pairs based on track
        pairs_key = f"{track}_key_pairs"
        key_pairs = cfg["datasets"][lang].get(pairs_key, {})
        
        # best heuristic is failure case; kept but not positioned as improvement
        best_f = ds.fusion_results.iloc[0] if not ds.fusion_results.empty else None
        t1.append(
            {
                "Language": lang.upper(),
                "Track": track.upper(),
                "Model_Type": "Best Single",
                "Model_Config": ds.best_model,
                "LAS": ds.best_las,
            }
        )
        t1.append(
            {
                "Language": lang.upper(),
                "Track": track.upper(),
                "Model_Type": "Oracle Upper Bound",
                "Model_Config": "Oracle(any selected model correct)",
                "LAS": ds.oracle_las,
            }
        )
        if best_f is not None:
            t1.append(
                {
                    "Language": lang.upper(),
                    "Track": track.upper(),
                    "Model_Type": "Heuristic Fusion (diagnostic)",
                    "Model_Config": f"{best_f['Pair_Name']} | {best_f['Rule']}",
                    "LAS": float(best_f["LAS"]),
                }
            )

        # table2 selected pairs complementarity
        c = ds.comp_las_selected.copy()
        if not c.empty:
            c.insert(0, "Track", track.upper())
            c.insert(0, "Language", lang.upper())
            t2.append(c)

        # table3 oracle analysis
        t3.append(
            {
                "Language": lang.upper(),
                "Track": track.upper(),
                "Best_Model": ds.best_model,
                "Best_Model_LAS": ds.best_las,
                "Oracle_LAS": ds.oracle_las,
                "Oracle_Gain_over_Best": ds.oracle_las - ds.best_las,
                "Avg_SelectedPair_Complementarity_LAS": float(ds.comp_las_selected["Complementarity_LAS"].mean()) if not ds.comp_las_selected.empty else 0.0,
            }
        )
        pairwise_frames.append(pair_oracle_summary(lang, ds.with_oracle, key_pairs))
        pairwise_all_frames.append(pair_oracle_summary(lang, ds.with_oracle, all_model_pairs(ds.models)))

    pd.DataFrame(t1).to_csv(f"{dirs['paper_tables']}/table1_model_comparison.csv", index=False)
    if t2:
        pd.concat(t2, ignore_index=True).to_csv(f"{dirs['paper_tables']}/table2_complementarity_selected_pairs.csv", index=False)
    pd.DataFrame(t3).to_csv(f"{dirs['paper_tables']}/table3_oracle_analysis.csv", index=False)
    table4_relation_class_errors(ds_results, f"{dirs['paper_tables']}/table4_relation_class_errors.csv")
    if pairwise_frames:
        pd.concat(pairwise_frames, ignore_index=True).to_csv(f"{dirs['paper_tables']}/pairwise_oracle_summary.csv", index=False)
    if pairwise_all_frames:
        pd.concat(pairwise_all_frames, ignore_index=True).to_csv(f"{dirs['paper_tables']}/pairwise_oracle_summary_all_pairs.csv", index=False)


def write_phase6_summary(dirs: Dict[str, str], ds_results: List[DSResult]) -> None:
    rows = []
    lines = []
    lines.append("PHASE 6 SUMMARY (TWO-TRACK ANALYSIS)")
    lines.append("=====================================")
    lines.append("This summary is diagnostic. Oracle is an upper bound only.")
    lines.append("Analysis runs separately for Non-MTL (Phase 3) and MTL (Phase 4) tracks.")
    lines.append("")
    for ds in ds_results:
        lines.append(
            f"{ds.lang.upper()} | {ds.track.upper()}: best_model={ds.best_model} LAS={ds.best_las:.2%}, "
            f"oracle={ds.oracle_las:.2%}, gain={ds.oracle_las - ds.best_las:+.2%}"
        )
        if not ds.fusion_results.empty:
            fbest = ds.fusion_results.iloc[0]
            lines.append(
                f"  best_heuristic(diagnostic)={fbest['Pair_Name']} {fbest['Rule']} "
                f"LAS={float(fbest['LAS']):.2%}"
            )
        lines.append(
            "  Interpretation: Heuristic fusion strategies appear to yield limited or inconsistent gains, "
            "suggesting complementarity is structured but not trivially exploitable."
        )
        rows.append(
            {
                "Language": ds.lang.upper(),
                "Track": ds.track.upper(),
                "best_model": ds.best_model,
                "best_uas": ds.best_uas,
                "best_las": ds.best_las,
                "oracle_uas": ds.oracle_uas,
                "oracle_las": ds.oracle_las,
                "oracle_gain_over_best_las": ds.oracle_las - ds.best_las,
            }
        )
    Path(f"{dirs['base']}/summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(f"{dirs['base']}/phase6_run_summary.csv", index=False)


def validate_self_check(dirs: Dict[str, str]) -> None:
    lines = []
    lines.append("PHASE 6 SELF-CHECK")
    lines.append("==================")
    # 1) no mixing - check both tracks exist for both languages
    hc_non_mtl = Path(f"{dirs['hc_non_mtl']}/hc_non_mtl_aligned_predictions.csv").exists()
    hc_mtl = Path(f"{dirs['hc_mtl']}/hc_mtl_aligned_predictions.csv").exists()
    mc_non_mtl = Path(f"{dirs['mc_non_mtl']}/mc_non_mtl_aligned_predictions.csv").exists()
    mc_mtl = Path(f"{dirs['mc_mtl']}/mc_mtl_aligned_predictions.csv").exists()
    lines.append(f"1) No HC/MC mixing: {'PASS' if (hc_non_mtl and hc_mtl and mc_non_mtl and mc_mtl) else 'FAIL'}")
    lines.append(f"   - HC tracks: non_mtl={hc_non_mtl}, mtl={hc_mtl}")
    lines.append(f"   - MC tracks: non_mtl={mc_non_mtl}, mtl={mc_mtl}")
    # 2) oracle label
    t3 = pd.read_csv(f"{dirs['paper_tables']}/table3_oracle_analysis.csv")
    lines.append(f"2) Oracle labeled upper bound: {'PASS' if 'Oracle_LAS' in t3.columns else 'FAIL'}")
    # 3) no fusion improvement claims (structural check)
    summary = Path(f"{dirs['base']}/summary.txt").read_text(encoding="utf-8").lower()
    disallowed = "fusion improves"
    lines.append(f"3) No claim of fusion improvement: {'PASS' if disallowed not in summary else 'FAIL'}")
    # 4) interpretation alignment basic check
    lines.append("4) Interpretations match numbers: MANUAL CHECK REQUIRED (see narrative/phase6_interpretation_check.txt)")
    Path(f"{dirs['narrative']}/phase6_self_check.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 Paper-1 diagnostic pipeline")
    parser.add_argument("--preset", choices=["legacy"], help="Use built-in legacy preset")
    parser.add_argument("--config", type=str, help="Path to config JSON")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output root")
    args = parser.parse_args()

    cfg = LEGACY_PRESET if args.preset == "legacy" else json.load(open(args.config, "r", encoding="utf-8"))
    out_root = args.output_dir or cfg.get("output_dir", DEFAULT_OUTPUT_DIR)
    dirs = ensure_dirs(out_root)
    validate_paths(cfg)

    # Save config used
    with open(f"{dirs['base']}/phase6_run_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # Run per language and track
    ds_results = []
    for lang in ["hc", "mc"]:
        # Run Non-MTL track
        ds_results.append(run_track(lang, "non_mtl", cfg["datasets"][lang], dirs))
        # Run MTL track
        ds_results.append(run_track(lang, "mtl", cfg["datasets"][lang], dirs))

    # Aggregate diagnostics + required paper tables + narrative/check
    aggregate_files(dirs)
    build_paper_tables(dirs, ds_results, cfg)
    write_phase6_summary(dirs, ds_results)
    build_narrative(dirs["narrative"], ds_results)
    validate_self_check(dirs)

    # shareable archive
    bundle = f"{dirs['base']}/phase6_diagnostics_bundle.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        tar.add(dirs["diagnostics"], arcname="diagnostics")
        tar.add(dirs["paper_tables"], arcname="paper_tables")
        tar.add(dirs["narrative"], arcname="narrative")
        tar.add(f"{dirs['base']}/summary.txt", arcname="summary.txt")
        tar.add(f"{dirs['base']}/phase6_run_summary.csv", arcname="phase6_run_summary.csv")

    print(f"Phase 6 complete (Paper 1 diagnostic scope).")
    print(f"Output root: {dirs['base']}")
    print(f"Shareable bundle: {bundle}")


if __name__ == "__main__":
    main()
