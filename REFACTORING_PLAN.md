# SuPar Creole Refactoring Plan

## Current State Analysis

### 21 Shell Scripts Pattern:
```
run_{action}_{language}_{variant}.sh

Actions: training, prediction, evaluation
Languages: hc (Haitian Creole), mc (Martinican Creole), fr+hc, fr+mc, hc+mc, mc+hc
Variants: baseline, enhanced
```

### What Each Script Does:
- **Training**: Calls `python3 -m supar.cmds.dep.biaffine train` with different data paths
- **Prediction**: Calls `python3 -m supar.cmds.dep.biaffine predict`
- **Evaluation**: Calls `python3 -m supar.cmds.dep.biaffine evaluate`

### Key Parameters That Vary:
1. Language/data combination (hc, mc, fr+hc, etc.)
2. Model paths
3. Data paths
4. BERT model variant (roberta, camembert, creoleval)
5. Random seeds
6. Baseline vs enhanced (baseline uses multiple seeds)

## Refactoring Strategy

### Option 1: Unified Python CLI (like QNLP)
```bash
python run.py train --language hc --variant baseline --model roberta
python run.py predict --language hc --model path/to/model
python run.py evaluate --language hc --model path/to/model
```

### Option 2: Keep Shell Scripts but Organize
- Move to scripts/ directory
- Create wrapper run.py that calls them
- Document which to use

### Option 3: Hybrid
- Create run.py for common cases
- Keep complex multi-seed scripts in scripts/

## Recommended Approach: Option 1

Create unified `run.py` similar to QNLP:
```python
python run.py train --language hc --seeds 1,2,3 --models roberta,camembert
python run.py predict --language hc --model models/hc/best_model
python run.py evaluate --language hc --predictions pred.conllu --gold data/test.conllu
```

## Implementation Steps

1. **Create run.py** - Unified CLI
2. **Archive shell scripts** - Move to scripts/archived/
3. **Update .gitignore** - Ignore results, logs, models
4. **Clean root directory** - Move .conllu files to data/
5. **Create sample data** - Small subset for testing
6. **Update README** - New usage patterns
7. **Test workflows** - Verify train/predict/evaluate work

## Files to Clean Up

Root directory cleanup:
- *.conllu → data/
- *.csv, *.png → results/
- run_*.sh → scripts/archived/ (keep as reference)
- supar_venv/ → Add to .gitignore

## New Structure
```
.
├── run.py                 # Unified CLI
├── configs/
├── data/
│   ├── hc/
│   ├── mc/
│   └── sample/           # NEW: test data
├── models/
├── predictions/
├── logs/
├── results/              # All CSV/PNG files
├── scripts/
│   └── archived/         # OLD shell scripts
└── supar/                # Library code
```
