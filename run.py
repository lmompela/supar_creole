#!/usr/bin/env python3
"""
SuPar Creole - Unified CLI for Dependency Parsing

Usage:
    python run.py train --language hc --variant baseline --model roberta
    python run.py predict --language hc --model models/hc/best_model
    python run.py evaluate --language hc --gold data/test.conllu --pred predictions/test.conllu
"""

import argparse
import subprocess
import sys
from pathlib import Path


# Language configurations
LANGUAGES = {
    'hc': {
        'name': 'Haitian Creole',
        'train': 'data/hc/seed{seed}/hc_train_seed{seed}.conllu',
        'dev': 'data/hc/seed{seed}/hc_dev_seed{seed}.conllu',
        'test': 'data/hc/hc_original_split_test.conllu',
        'embed': 'embeddings/cc.ht.100.vec'
    },
    'mc': {
        'name': 'Martinican Creole',
        'train': 'data/mc/seed{seed}/mc_train_seed{seed}.conllu',
        'dev': 'data/mc/seed{seed}/mc_dev_seed{seed}.conllu',
        'test': 'data/mc/mc_original_split_test.conllu',
        'embed': 'embeddings/cc.ht.100.vec'
    },
    'fr+hc': {
        'name': 'French + Haitian Creole',
        'train': 'data/fr_hc/fr_hc_train.conllu',
        'dev': 'data/fr_hc/fr_hc_dev.conllu',
        'test': 'data/hc/hc_original_split_test.conllu',
        'embed': 'embeddings/cc.ht.100.vec'
    },
    'fr+mc': {
        'name': 'French + Martinican Creole',
        'train': 'data/fr_mc/fr_mc_train.conllu',
        'dev': 'data/fr_mc/fr_mc_dev.conllu',
        'test': 'data/mc/mc_original_split_test.conllu',
        'embed': 'embeddings/cc.ht.100.vec'
    },
}

# BERT model configurations
BERT_MODELS = {
    'roberta': 'roberta-large',
    'camembert': 'camembert-base',
    'camembert2': 'almanach/camembertv2-base',
    'creoleval': 'lgrobol/xlm-r-CreoleEval_all'
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
    
    # Prepare paths
    model_dir = Path(f"models/{args.language}_{args.variant}/{args.model}")
    log_dir = Path(f"logs/{args.language}")
    
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Training {args.language} - {args.model} - seed {seed}")
        print(f"{'='*60}\n")
        
        model_path = model_dir / f"model_seed{seed}"
        log_file = log_dir / f"{args.model}_seed{seed}_train.log"
        
        cmd = [
            'python3', '-m', 'supar.cmds.dep.biaffine', 'train',
            '-b', '-d', str(args.device),
            '-c', args.config,
            '-p', str(model_path),
            '--train', lang_config['train'].format(seed=seed),
            '--dev', lang_config['dev'].format(seed=seed),
            '--test', lang_config['test'],
            '--feat', 'tag',
            '--encoder', 'bert',
            '--embed', lang_config['embed'],
            '--bert', bert_model,
            '--tree'
        ]
        
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
        '-c', args.config,
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
    """Evaluate predictions against gold standard."""
    print(f"\n{'='*60}")
    print(f"Evaluating predictions")
    print(f"Gold: {args.gold}")
    print(f"Pred: {args.pred}")
    print(f"{'='*60}\n")
    
    cmd = [
        'python3', '-m', 'supar.cmds.dep.biaffine', 'evaluate',
        '-d', str(args.device),
        '--data', args.gold,
        '--pred', args.pred,
        '--tree'
    ]
    
    if args.model:
        cmd.extend(['-p', args.model])
    
    returncode = subprocess.run(cmd).returncode
    
    if returncode != 0:
        print(f"\n✗ Evaluation failed")
        sys.exit(1)


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
        choices=['baseline', 'enhanced'],
        default='enhanced',
        help='Training variant'
    )
    train_parser.add_argument(
        '--model', '-m',
        choices=list(BERT_MODELS.keys()) + ['custom'],
        default='roberta',
        help='BERT model to use'
    )
    train_parser.add_argument(
        '--seeds',
        type=lambda s: [int(x) for x in s.split(',')],
        default='1',
        help='Random seeds (comma-separated, e.g., 1,2,3)'
    )
    train_parser.add_argument(
        '--config', '-c',
        default='configs/config_experiment.config',
        help='Config file path'
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
        '--config', '-c',
        default='configs/config_experiment.config',
        help='Config file path'
    )
    predict_parser.add_argument(
        '--device', '-d',
        type=int,
        default=0,
        help='GPU device ID'
    )
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate predictions')
    evaluate_parser.add_argument(
        '--gold', '-g',
        required=True,
        help='Gold standard data file'
    )
    evaluate_parser.add_argument(
        '--pred', '-p',
        required=True,
        help='Prediction file to evaluate'
    )
    evaluate_parser.add_argument(
        '--model', '-m',
        help='Model path (optional)'
    )
    evaluate_parser.add_argument(
        '--device', '-d',
        type=int,
        default=0,
        help='GPU device ID'
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
