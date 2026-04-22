#!/usr/bin/env python3
"""
SuPar Creole - Unified CLI for Dependency Parsing

Experiment naming convention (output directories):
    models/{data_config}__{lm}__{feat_config}__{embed}/model_seed{N}
    logs/{data_config}/{lm}__{feat_config}__{embed}_seed{N}.log
    predictions/{data_config}/{lm}__{feat_config}__{embed}_seed{N}.conllu

Data configs:
    hc           - Haitian Creole only           → test: HC test set
    mc           - Martinican Creole only         → test: MC test set
    hc+mc        - HC augments MC training data   → test: MC test set
    mc+hc        - MC augments HC training data   → test: HC test set
    fr+hc        - FR augments HC training data   → test: HC test set
    fr+mc        - FR augments MC training data   → test: MC test set

Feat configs:
    bert-enc          - encoder=bert  (fine-tune BERT as full encoder; --embed ignored)
    lstm-tag          - encoder=lstm, feat=tag
    lstm-char         - encoder=lstm, feat=char
    lstm-bert         - encoder=lstm, feat=bert  (BERT as frozen feature extractor)
    lstm-tag-char-bert- encoder=lstm, feat=tag char bert

Embed keys (only used when encoder=lstm):
    ht-ft   - Haitian Creole fastText  (embeddings/cc.ht.100.vec)
    fr-ft   - French fastText          (embeddings/cc.fr.300.vec)
    glove   - GloVE 6B 100d            (embeddings/glove.6B.100d.txt)
    none    - random initialisation    (no --embed flag passed)

Usage:
    python run.py train --language hc --variant baseline --model roberta
    python run.py train --language hc --variant baseline --model camembert --feat-config lstm-tag --embed-key ht-ft
    python run.py predict --model models/hc__roberta__bert-enc__n-a/model_seed1 --data data/hc/hc_original_split_test.conllu --output predictions/test.conllu
    python run.py evaluate --gold data/hc/hc_original_split_test.conllu --pred predictions/test.conllu
"""

import argparse
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Language configurations
# Naming: X+Y means "use X to augment Y training data; test on Y"
# ---------------------------------------------------------------------------
LANGUAGES = {
    'hc': {
        'name': 'Haitian Creole',
        'train': 'data/hc/seed{seed}/hc_train_seed{seed}.conllu',
        'dev': 'data/hc/seed{seed}/hc_dev_seed{seed}.conllu',
        'test': 'data/hc/hc_original_split_test.conllu',
    },
    'mc': {
        'name': 'Martinican Creole',
        'train': 'data/mc/seed{seed}/mc_train_seed{seed}.conllu',
        'dev': 'data/mc/seed{seed}/mc_dev_seed{seed}.conllu',
        'test': 'data/mc/mc_original_split_test.conllu',
    },
    # Concatenation experiments: X+Y = augment Y with X, evaluate on Y
    'hc+mc': {
        'name': 'HC augments MC (train on HC+MC, test on MC)',
        'train': 'data/hc+mc/seed{seed}/hc+mc_train_seed{seed}.conllu',
        'dev': 'data/hc+mc/seed{seed}/hc+mc_dev_seed{seed}.conllu',
        'test': 'data/mc/mc_original_split_test.conllu',
    },
    'mc+hc': {
        'name': 'MC augments HC (train on MC+HC, test on HC)',
        'train': 'data/mc+hc/seed{seed}/mc+hc_train_seed{seed}.conllu',
        'dev': 'data/mc+hc/seed{seed}/mc+hc_dev_seed{seed}.conllu',
        'test': 'data/hc/hc_original_split_test.conllu',
    },
    'fr+hc': {
        'name': 'FR augments HC (train on FR+HC, test on HC)',
        'train': 'data/fr+hc/seed{seed}/fr+hc_train_seed{seed}.conllu',
        'dev': 'data/fr+hc/seed{seed}/fr+hc_dev_seed{seed}.conllu',
        'test': 'data/hc/hc_original_split_test.conllu',
    },
    'fr+mc': {
        'name': 'FR augments MC (train on FR+MC, test on MC)',
        'train': 'data/fr+mc/seed{seed}/fr+mc_train_seed{seed}.conllu',
        'dev': 'data/fr+mc/seed{seed}/fr+mc_dev_seed{seed}.conllu',
        'test': 'data/mc/mc_original_split_test.conllu',
    },
}

# ---------------------------------------------------------------------------
# BERT / language model configurations
# ---------------------------------------------------------------------------
BERT_MODELS = {
    'roberta': 'roberta-large',
    'camembert': 'camembert-base',
    'camembert2': 'almanach/camembertv2-base',
    'creoleval': 'lgrobol/xlm-r-CreoleEval_all'
}

# ---------------------------------------------------------------------------
# Feature / encoder configurations
# Each entry: (encoder, feat_list, config_file)
# 'bert-enc' fine-tunes BERT as the full encoder; --embed is ignored.
# 'lstm-*' use a BiLSTM encoder with the specified features.
# ---------------------------------------------------------------------------
FEAT_CONFIGS = {
    'bert-enc':           ('bert', [],                    'configs/config_bert_enc.config'),
    'lstm-tag':           ('lstm', ['tag'],               'configs/config_lstm.config'),
    'lstm-char':          ('lstm', ['char'],              'configs/config_lstm.config'),
    'lstm-bert':          ('lstm', ['bert'],              'configs/config_lstm.config'),
    'lstm-tag-char-bert': ('lstm', ['tag', 'char', 'bert'], 'configs/config_lstm.config'),
}

# ---------------------------------------------------------------------------
# Word embedding configurations (only used when encoder=lstm)
# Each entry: (embed_path_or_key, n_pretrained_dim)
# 'none' passes --embed '' so SuPar skips pretrained init (random word vectors only).
# 'glove' uses SuPar's auto-download key; dim=100 matches SuPar's n_embed default.
# fastText files are 300d and require --n-pretrained 300 so SuPar builds the model correctly.
# ---------------------------------------------------------------------------
EMBEDDINGS = {
    'ht-ft':      ('embeddings/cc.ht.100.vec', 100),    # Haitian Creole fastText 100d (PCA-reduced, 96.1% var)
    'fr-ft':      ('embeddings/cc.fr.100.vec', 100),    # French fastText 100d (PCA-reduced, 60.8% var)
    'en-ft':      ('embeddings/cc.en.100.vec', 100),    # English fastText 100d (PCA-reduced, 59.6% var)
    'ht-ft-300':  ('embeddings/cc.ht.300.vec', 300),    # Haitian Creole fastText 300d (original)
    'fr-ft-300':  ('embeddings/cc.fr.300.vec', 300),    # French fastText 300d (original)
    'en-ft-300':  ('embeddings/cc.en.300.vec', 300),    # English fastText 300d (original)
    'glove':      ('glove-6b-100', 100),                # GloVE 6B 100d (SuPar auto-download)
    'none':       ('', None),                           # random init — skip pretrained
}


def run_command(cmd, log_file=None):
    """Execute a command and optionally log output."""
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'w') as f:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in proc.stdout:
                line_str = line.decode()
                print(line_str, end='')
                f.write(line_str)
            proc.wait()
            return proc.returncode
    else:
        return subprocess.run(cmd).returncode


def train(args):
    """Train a dependency parser."""
    lang_config = LANGUAGES[args.language]
    bert_model = BERT_MODELS.get(args.model, args.model)
    encoder, feats, default_config = FEAT_CONFIGS[args.feat_config]
    embed_path, embed_dim = EMBEDDINGS[args.embed_key]

    # Config file: explicit --config overrides the feat-config default
    config_file = args.config if args.config else default_config

    # Output directory encodes all experiment dimensions for traceability
    # embed dimension uses 'n-a' when encoder=bert (embed is irrelevant)
    embed_label = 'n-a' if encoder == 'bert' else args.embed_key
    run_label = f"{args.language}__{args.model}__{args.feat_config}__{embed_label}"
    model_dir = Path(f"models/{run_label}")
    log_dir = Path(f"logs/{args.language}")

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Language  : {lang_config['name']}")
    print(f"LM        : {args.model} ({bert_model})")
    print(f"Feat cfg  : {args.feat_config}  (encoder={encoder}, feats={feats or 'none'})")
    print(f"Embed     : {embed_label}")
    print(f"Seeds     : {args.seeds}")
    print(f"Config    : {config_file}")
    print(f"Output dir: {model_dir}")
    print(f"{'='*60}\n")

    for seed in args.seeds:
        print(f"--- seed {seed} ---")

        model_path = model_dir / f"model_seed{seed}"
        log_file = log_dir / f"{args.model}__{args.feat_config}__{embed_label}_seed{seed}.log"

        cmd = [
            'python3', '-m', 'supar.cmds.dep.biaffine', 'train',
            '-b', '-d', str(args.device),
            '-c', config_file,
            '-p', str(model_path),
            '--train', lang_config['train'].format(seed=seed),
            '--dev', lang_config['dev'].format(seed=seed),
            '--test', lang_config['test'],
            '--encoder', encoder,
            '--bert', bert_model,
            '--seed', str(seed),
            '--tree'
        ]

        if feats:
            cmd += ['--feat'] + feats

        # Word embeddings: always pass --embed for lstm encoder so SuPar
        # doesn't silently fall back to its glove-6b-100 default.
        # Empty string ('none') tells SuPar to skip pretrained init.
        # n_pretrained is auto-synced in parser.py build() from the embedding dim.
        if encoder == 'lstm':
            cmd += ['--embed', embed_path]

        returncode = run_command(cmd, log_file if not args.no_log else None)

        if returncode != 0:
            print(f"✗ Training failed for seed {seed}")
            if not args.continue_on_error:
                sys.exit(1)
        else:
            print(f"✓ Training completed for seed {seed}")


def predict(args):
    """Generate predictions using a trained model."""
    model_path = Path(args.model)
    
    if not model_path.exists():
        print(f"✗ Model not found: {model_path}")
        sys.exit(1)
    
    pred_dir = Path(args.output).parent
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generating predictions")
    print(f"Model: {model_path}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")
    
    cmd = [
        'python3', '-m', 'supar.cmds.dep.biaffine', 'predict',
        '-d', str(args.device),
        '-p', str(model_path),
        '--pred', args.output,
        '--data', args.data,
        '--prob',
        '--tree'
    ]
    
    returncode = subprocess.run(cmd).returncode
    
    if returncode == 0:
        print(f"\n✓ Predictions saved to: {args.output}")
    else:
        print(f"\n✗ Prediction failed")
        sys.exit(1)


def evaluate(args):
    """Evaluate predictions against gold standard using CoNLL-U eval script."""
    import json
    import re

    print(f"\n{'='*60}")
    print(f"Evaluating predictions")
    print(f"Gold: {args.gold}")
    print(f"Pred: {args.pred}")
    print(f"{'='*60}\n")

    cmd = [
        'python3', 'scripts/conll17_ud_eval.py',
        '-v',
        args.gold,
        args.pred
    ]

    # Capture output, print it to stdout, and parse for JSON — single run
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout, end='')
    if result.stderr:
        print(result.stderr, end='', file=sys.stderr)

    if result.returncode != 0:
        print(f"\n✗ Evaluation failed")
        sys.exit(1)

    metrics = {}
    for line in result.stdout.splitlines():
        for metric in ('UAS', 'LAS', 'CLAS'):
            if re.match(rf'^{metric}\s*\|', line):
                parts = [p.strip() for p in line.split('|')]
                try:
                    metrics[metric] = float(parts[3])  # F1 Score column
                except (ValueError, IndexError):
                    pass

    if metrics:
        json_path = Path(args.pred).with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump({'gold': args.gold, 'pred': args.pred, 'metrics': metrics}, f, indent=2)
        print(f"✓ Metrics saved to {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description='SuPar Creole - Unified CLI for dependency parsing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a dependency parser')
    train_parser.add_argument(
        '--language', '-l',
        required=True,
        choices=list(LANGUAGES.keys()),
        help='Language or language combination'
    )
    train_parser.add_argument(
        '--variant', '-v',
        default='baseline',
        help='(Unused in output path; kept for backwards-compat notes only)'
    )
    train_parser.add_argument(
        '--model', '-m',
        choices=list(BERT_MODELS.keys()) + ['custom'],
        default='roberta',
        help='BERT model to use'
    )
    train_parser.add_argument(
        '--feat-config', '-f',
        choices=list(FEAT_CONFIGS.keys()),
        default='bert-enc',
        help=(
            'Feature/encoder configuration. '
            'bert-enc: fine-tune BERT as full encoder (--embed ignored). '
            'lstm-*: BiLSTM encoder with the named feature(s).'
        )
    )
    train_parser.add_argument(
        '--embed-key', '-e',
        choices=list(EMBEDDINGS.keys()),
        default='ht-ft',
        help='Word embedding initialisation (only used when --feat-config is lstm-*)'
    )
    train_parser.add_argument(
        '--seeds',
        type=lambda s: [int(x) for x in s.split(',')],
        default='1',
        help='Random seeds (comma-separated, e.g., 1,2,3)'
    )
    train_parser.add_argument(
        '--config', '-c',
        default=None,
        help='Override config file path (default: auto-selected from feat-config)'
    )
    train_parser.add_argument(
        '--device', '-d',
        type=int,
        default=0,
        help='GPU device ID'
    )
    train_parser.add_argument(
        '--no-log',
        action='store_true',
        help='Do not save logs to file'
    )
    train_parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue training other seeds if one fails'
    )
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Generate predictions')
    predict_parser.add_argument(
        '--model', '-m',
        required=True,
        help='Path to trained model'
    )
    predict_parser.add_argument(
        '--data', '-i',
        required=True,
        help='Input data file (CoNLL-U format)'
    )
    predict_parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output prediction file'
    )
    predict_parser.add_argument(
        '--device', '-d',
        type=int,
        default=0,
        help='GPU device ID'
    )
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate predictions against gold CoNLL-U file')
    evaluate_parser.add_argument(
        '--gold', '-g',
        required=True,
        help='Gold standard CoNLL-U file'
    )
    evaluate_parser.add_argument(
        '--pred', '-p',
        required=True,
        help='Predicted CoNLL-U file to evaluate'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    if args.command == 'train':
        train(args)
    elif args.command == 'predict':
        predict(args)
    elif args.command == 'evaluate':
        evaluate(args)


if __name__ == '__main__':
    main()
