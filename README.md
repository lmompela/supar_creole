# SuPar Creole - Dependency Parsing for Creole Languages

Unified CLI for single-task dependency parsing experiments on low-resource Creole data.

## Supported data configs

- `hc` (Haitian Creole only)
- `mc` (Martinican Creole only)
- `hc+mc` (HC augments MC; test on MC)
- `mc+hc` (MC augments HC; test on HC)
- `fr+hc` (French augments HC; test on HC)
- `fr+mc` (French augments MC; test on MC)

Convention: `X+Y` means **augment Y with X**, evaluate on Y test data.

## Supported model keys

- `roberta` (`roberta-large`)
- `camembert` (`camembert-base`)
- `camembert2` (`almanach/camembertv2-base`)
- `creoleval` (`lgrobol/xlm-r-CreoleEval_all`)

## Installation

```bash
git clone <your-repo-url>
cd supar_creole
python3 -m venv supar_venv
source supar_venv/bin/activate
pip install -U supar
pip install -e .
```

## Embeddings

Used only when `--feat-config` is LSTM-based (`lstm-*`):

- `ht-ft` -> `embeddings/cc.ht.100.vec`
- `fr-ft` -> `embeddings/cc.fr.100.vec`
- `en-ft` -> `embeddings/cc.en.100.vec`
- `ht-ft-300`, `fr-ft-300`, `en-ft-300` -> 300d fastText files
- `glove` -> `glove-6b-100` (SuPar registry key)
- `none` -> random initialization

For `bert-enc`, `--embed-key` is ignored.

## Quick start

### 1. Train

```bash
# BERT encoder (default feat-config)
python run.py train --language hc --model creoleval --seeds 1,2,3

# LSTM encoder with explicit feature/embedding setup
python run.py train \
  --language fr+hc \
  --model creoleval \
  --feat-config lstm-tag-char-bert \
  --embed-key glove \
  --seeds 1,2,3
```

### 2. Predict

```bash
python run.py predict \
  --model models/hc__creoleval__bert-enc__n-a/model_seed1 \
  --data data/hc/hc_original_split_test.conllu \
  --output predictions/hc_seed1.conllu
```

### 3. Evaluate

```bash
python run.py evaluate \
  --gold data/hc/hc_original_split_test.conllu \
  --pred predictions/hc_seed1.conllu
```

This prints CoNLL-U evaluation and saves a JSON sidecar next to `--pred`.

## CLI reference

### `train`

```bash
python run.py train --language {hc,mc,hc+mc,mc+hc,fr+hc,fr+mc} [OPTIONS]
```

Key options:

- `--model {roberta,camembert,camembert2,creoleval,custom}`
- `--feat-config {bert-enc,lstm-tag,lstm-char,lstm-bert,lstm-tag-char-bert}`
- `--embed-key {ht-ft,fr-ft,en-ft,ht-ft-300,fr-ft-300,en-ft-300,glove,none}`
- `--seeds 1,2,3`
- `--config` (optional override; otherwise auto-selects:
  - `configs/config_bert_enc.config` for `bert-enc`
  - `configs/config_lstm.config` for `lstm-*`)
- `--device`, `--no-log`, `--continue-on-error`

Note: `--variant` is kept for backward compatibility but is not used for output path naming.

### `predict`

```bash
python run.py predict --model <model_path> --data <input.conllu> --output <pred.conllu> [--device 0]
```

### `evaluate`

```bash
python run.py evaluate --gold <gold.conllu> --pred <pred.conllu>
```

## Output naming convention

Training outputs are organized as:

- Models: `models/{data_config}__{lm}__{feat_config}__{embed}/model_seed{N}`
- Logs: `logs/{data_config}/{lm}__{feat_config}__{embed}_seed{N}.log`
- Predictions (recommended): `predictions/{data_config}/{lm}__{feat_config}__{embed}_seed{N}.conllu`

For `bert-enc`, embed segment is `n-a`.

## Project structure

```text
supar_creole/
├── run.py
├── configs/
│   ├── config_bert_enc.config
│   └── config_lstm.config
├── data/
│   ├── hc/
│   ├── mc/
│   ├── hc+mc/
│   ├── mc+hc/
│   ├── fr+hc/
│   └── fr+mc/
├── embeddings/   # local only (gitignored)
├── models/       # local only (often symlinked)
├── logs/         # local only
├── predictions/  # local only
├── results/
└── scripts/
```

## License

MIT License
