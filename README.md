# SuPar Creole - Dependency Parsing for Creole Languages

This repository provides a unified CLI for training and evaluating dependency parsers for Creole languages using the [SuPar](https://github.com/yzhangcs/parser) toolkit.

## Supported Languages

- **Haitian Creole** (hc)
- **Martinican Creole** (mc)  
- **French + Haitian Creole** (fr+hc)
- **French + Martinican Creole** (fr+mc)

## Supported Models

- **RoBERTa Large** (`roberta`)
- **CamemBERT Base** (`camembert`)
- **CamemBERT v2 Base** (`camembert2`)
- **XLM-R CreoleEval** (`creoleval`)

## Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd supar_creole
```

### 2. Create virtual environment
```bash
python3 -m venv supar_venv
source supar_venv/bin/activate
```

### 3. Install SuPar
```bash
pip install -U supar
```

Or install from this repository:
```bash
pip install -e .
```

### 4. Download Embeddings

The FastText embeddings are required but not included in the repository (103MB file exceeds GitHub limit).

**Option 1: Download from FastText**
```bash
cd embeddings
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ht.300.bin.gz
gunzip cc.ht.300.bin.gz
```

**Option 2: Use the 100-dimension vectors (recommended)**
```bash
# Contact repository owner or use your own embeddings
# Place embeddings/cc.ht.100.vec in the embeddings/ directory
```

**Note:** The scripts reference `embeddings/cc.ht.100.vec`. You can also modify `run.py` to use different embeddings if needed.

### Requirements
- Python >= 3.8
- PyTorch >= 1.8
- Transformers >= 4.0
- SuPar >= 1.1.0
- FastText embeddings (see above)

## Quick Start

### Training a Model

Train a model for Haitian Creole using RoBERTa:
```bash
python run.py train \
  --language hc \
  --variant enhanced \
  --model roberta \
  --seeds 1,2,3
```

Train with multiple languages (French + Haitian):
```bash
python run.py train \
  --language fr+hc \
  --model camembert \
  --seeds 1
```

### Making Predictions

Generate predictions on test data:
```bash
python run.py predict \
  --model models/hc_enhanced/roberta/model_seed1 \
  --data data/hc/hc_original_split_test.conllu \
  --output predictions/hc_test_predictions.conllu
```

### Evaluating a Model

Evaluate predictions against gold standard:
```bash
python run.py evaluate \
  --gold data/hc/hc_original_split_test.conllu \
  --pred predictions/hc_test_predictions.conllu
```

## Usage Details

### Training Command

```bash
python run.py train [OPTIONS]
```

**Required Options:**
- `--language, -l`: Language code (hc, mc, fr+hc, fr+mc)

**Optional Options:**
- `--variant, -v`: Training variant (baseline, enhanced) [default: enhanced]
- `--model, -m`: BERT model (roberta, camembert, camembert2, creoleval) [default: roberta]
- `--seeds`: Comma-separated random seeds (e.g., 1,2,3) [default: 1]
- `--config, -c`: Config file path [default: configs/config_experiment.config]
- `--device, -d`: GPU device ID [default: 0]
- `--no-log`: Don't save logs to file
- `--continue-on-error`: Continue with other seeds if one fails

**Output:**
- Model: `models/{language}_{variant}/{model}/model_seed{N}`
- Logs: `logs/{language}/{model}_seed{N}_train.log`

### Prediction Command

```bash
python run.py predict [OPTIONS]
```

**Required Options:**
- `--model, -m`: Path to trained model
- `--data, -i`: Input data file (CoNLL-U format)
- `--output, -o`: Output prediction file

**Optional Options:**
- `--config, -c`: Config file path [default: configs/config_experiment.config]
- `--device, -d`: GPU device ID [default: 0]

### Evaluation Command

```bash
python run.py evaluate [OPTIONS]
```

**Required Options:**
- `--gold, -g`: Gold standard data file
- `--pred, -p`: Prediction file to evaluate

**Optional Options:**
- `--model, -m`: Model path (optional)
- `--device, -d`: GPU device ID [default: 0]

## Project Structure

```
supar_creole/
├── run.py                    # Unified CLI interface
├── configs/                  # Configuration files
├── data/                     # Training/test data
│   ├── hc/                  # Haitian Creole
│   ├── mc/                  # Martinican Creole
│   ├── fr_hc/               # French + Haitian
│   └── fr_mc/               # French + Martinican
├── embeddings/               # FastText embeddings
├── models/                   # Trained models (gitignored)
├── logs/                     # Training logs (gitignored)
├── predictions/              # Prediction outputs (gitignored)
├── results/                  # Evaluation results (gitignored)
└── scripts/                  # Utility scripts
    └── archived/            # Legacy shell scripts (reference)
```

## Migration from Shell Scripts

If you were previously using the shell scripts (`run_training_*.sh`, etc.), here's how to migrate:

**Old way:**
```bash
./run_training_hc_baseline_enhanced.sh
```

**New way:**
```bash
python run.py train --language hc --variant enhanced
```

**Benefits:**
- Single interface for all languages/models
- Better error handling
- Easier parameter customization
- Consistent logging
- No need to edit scripts for different configurations

The original shell scripts are preserved in `scripts/archived/` for reference.

## Examples

### Train multiple models
```bash
# Train all models for Haitian Creole
for model in roberta camembert camembert2 creoleval; do
  python run.py train --language hc --model $model --seeds 1,2,3
done
```

### Run full pipeline
```bash
# 1. Train
python run.py train --language hc --model roberta --seeds 1

# 2. Predict
python run.py predict \
  --model models/hc_enhanced/roberta/model_seed1 \
  --data data/hc/hc_original_split_test.conllu \
  --output predictions/hc_roberta_test.conllu

# 3. Evaluate
python run.py evaluate \
  --gold data/hc/hc_original_split_test.conllu \
  --pred predictions/hc_roberta_test.conllu
```

## Data Format

All data files should be in CoNLL-U format. Example:

```
1       Li      li      PRON    _       _       2       nsubj   _       _
2       manje   manje   VERB    _       _       0       root    _       _
3       pen     pen     NOUN    _       _       2       obj     _       _
4       .       .       PUNCT   _       _       2       punct   _       _
```

## Citation

If you use this code or models, please cite:

```bibtex
@inproceedings{zhang-etal-2020-efficient,
    title = "Efficient Second-Order {T}ree{CRF} for Neural Dependency Parsing",
    author = "Zhang, Yu and Li, Zhenghua and Zhang, Min",
    booktitle = "Proceedings of ACL",
    year = "2020",
    url = "https://www.aclweb.org/anthology/2020.acl-main.302",
    pages = "3295--3305"
}
```

## License

MIT License

## Upstream SuPar Documentation

For more information about the SuPar toolkit, see [README.upstream.md](README.upstream.md) or visit the [official SuPar repository](https://github.com/yzhangcs/parser).
