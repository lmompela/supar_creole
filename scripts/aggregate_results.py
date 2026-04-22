"""
Aggregate evaluation results across seeds for all configs in a predictions directory.
Reads pre-saved .json files produced by `run.py evaluate` (one per prediction file).

Usage:
  # Aggregate all configs for a language (HC or MC):
  python3 scripts/aggregate_results.py --lang hc
  python3 scripts/aggregate_results.py --lang mc

  # Single config, all seeds (quick check):
  python3 scripts/aggregate_results.py --lang hc --config "fr+hc__creoleval__lstm-tag-char-bert__glove"

  # Save output to a file (shows per-seed detail + averages):
  python3 scripts/aggregate_results.py --lang hc --seeds 1,2,3 --output results/hc_phase3.txt

  # Only specific seeds:
  python3 scripts/aggregate_results.py --lang hc --seeds 1,2,3
"""

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

GOLD_FILES = {
    'hc': 'data/hc/hc_original_split_test.conllu',
    'mc': 'data/mc/mc_original_split_test.conllu',
}


def extract_config_and_seed(filename: str):
    """Split e.g. 'fr+hc__creoleval__lstm-tag-char-bert__glove_seed2' into (config, seed)."""
    m = re.match(r'^(.+)_seed(\d+)$', Path(filename).stem)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def fmt_mean(vals):
    if not vals:
        return '     —     '
    if len(vals) == 1:
        return f'   {vals[0]:6.2f}    '
    mean = statistics.mean(vals)
    std = statistics.stdev(vals)
    return f'{mean:5.2f} ± {std:.2f}'


def build_report(lang, configs, gold_file):
    lines = []
    W = 92

    # ── Per-config detail blocks ─────────────────────────────────────────────
    lines.append('')
    lines.append('=' * W)
    lines.append(f'  PER-SEED DETAIL — {lang.upper()} | Gold: {gold_file}')
    lines.append('=' * W)

    sorted_configs = sorted(
        configs.items(),
        key=lambda x: -statistics.mean(v.get('LAS', 0) for v in x[1].values())
    )

    for config, seed_data in sorted_configs:
        seeds = sorted(seed_data.keys())
        lines.append('')
        lines.append(f'  Config: {config}')
        lines.append(f"  {'':>10}  {'UAS':>8}  {'LAS':>8}  {'CLAS':>8}")
        lines.append(f"  {'-'*40}")

        uas_vals, las_vals, clas_vals = [], [], []
        for s in seeds:
            m = seed_data[s]
            uas  = m.get('UAS',  None)
            las  = m.get('LAS',  None)
            clas = m.get('CLAS', None)
            uas_str  = f'{uas:8.2f}'  if uas  is not None else '       —'
            las_str  = f'{las:8.2f}'  if las  is not None else '       —'
            clas_str = f'{clas:8.2f}' if clas is not None else '       —'
            lines.append(f"  {'seed ' + str(s):>10}  {uas_str}  {las_str}  {clas_str}")
            if uas  is not None: uas_vals.append(uas)
            if las  is not None: las_vals.append(las)
            if clas is not None: clas_vals.append(clas)

        if len(seeds) > 1:
            lines.append(f"  {'-'*40}")
            def _avg(vals):
                return f'{statistics.mean(vals):8.2f}' if vals else '       —'
            def _std(vals):
                return f'± {statistics.stdev(vals):.2f}' if len(vals) > 1 else ''
            lines.append(
                f"  {'mean':>10}  {_avg(uas_vals)}  {_avg(las_vals)}  {_avg(clas_vals)}"
            )
            lines.append(
                f"  {'std':>10}  {_std(uas_vals):>8}  {_std(las_vals):>8}  {_std(clas_vals):>8}"
            )

    # ── Summary table ────────────────────────────────────────────────────────
    lines.append('')
    lines.append('=' * W)
    lines.append(f'  SUMMARY TABLE — {lang.upper()} | Gold: {gold_file}')
    lines.append('=' * W)
    lines.append(f"{'Config':<55} {'Seeds':>8}  {'UAS':>13}  {'LAS':>13}  {'CLAS':>13}")
    lines.append('-' * W)

    for config, seed_data in sorted_configs:
        seeds = sorted(seed_data.keys())
        seed_str  = '+'.join(str(s) for s in seeds)
        uas_vals  = [seed_data[s]['UAS']  for s in seeds if 'UAS'  in seed_data[s]]
        las_vals  = [seed_data[s]['LAS']  for s in seeds if 'LAS'  in seed_data[s]]
        clas_vals = [seed_data[s]['CLAS'] for s in seeds if 'CLAS' in seed_data[s]]
        lines.append(
            f'{config:<55} {seed_str:>8}  {fmt_mean(uas_vals):>13}'
            f'  {fmt_mean(las_vals):>13}  {fmt_mean(clas_vals):>13}'
        )

    lines.append('=' * W)
    lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Aggregate multi-seed evaluation results.')
    parser.add_argument('--lang', '-l', required=True, choices=['hc', 'mc'],
                        help='Language: hc or mc')
    parser.add_argument('--config', '-c', default=None,
                        help='Filter to a specific config name (substring match)')
    parser.add_argument('--pred-dir', default=None,
                        help='Override predictions directory')
    parser.add_argument('--seeds', default=None,
                        help='Only include these seeds (comma-separated, e.g. 1,2,3)')
    parser.add_argument('--output', '-o', default=None,
                        help='Save report to this file (also printed to stdout)')
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir or f'predictions/{args.lang}')
    seed_filter = set(int(s) for s in args.seeds.split(',')) if args.seeds else None

    if not pred_dir.exists():
        print(f'Error: predictions directory not found: {pred_dir}', file=sys.stderr)
        sys.exit(1)

    # Group JSON result files by config
    configs = defaultdict(dict)  # config -> {seed: metrics_dict}
    missing_json = []

    for json_file in sorted(pred_dir.glob('*.json')):
        config, seed = extract_config_and_seed(json_file.name)
        if config is None:
            continue
        if args.config and args.config not in config:
            continue
        if seed_filter and seed not in seed_filter:
            continue
        with open(json_file) as f:
            data = json.load(f)
        configs[config][seed] = data.get('metrics', {})

    # Warn about any prediction files that have no JSON yet
    for pred_file in sorted(pred_dir.glob('*.conllu')):
        config, seed = extract_config_and_seed(pred_file.name)
        if config is None:
            continue
        if args.config and args.config not in config:
            continue
        if seed_filter and seed not in seed_filter:
            continue
        if not pred_file.with_suffix('.json').exists():
            missing_json.append(pred_file.name)

    if missing_json:
        print("⚠  No JSON found for (run `python3 run.py evaluate` on these first):")
        for f in missing_json:
            print(f"   {f}")
        print()

    if not configs:
        print(f'No result JSON files found in {pred_dir}')
        sys.exit(0)

    report = build_report(args.lang, configs, GOLD_FILES[args.lang])
    print(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report)
        print(f'Report saved to: {out_path}')


if __name__ == '__main__':
    main()

