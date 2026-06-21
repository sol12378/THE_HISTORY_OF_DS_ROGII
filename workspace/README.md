# ROGII Wellbore Geology Prediction

Kaggle competition workspace for ROGII - Wellbore Geology Prediction.

## Layout

- `data/raw/`: immutable Kaggle downloads
- `data/interim/`: temporary preprocessing outputs
- `data/processed/`: model-ready datasets
- `data/folds/`: saved CV split files
- `src/rogii/`: reusable package code
- `scripts/`: command line entrypoints
- `experiments/`: one directory per experiment
- `outputs/`: logs, models, figures, predictions, blends
- `reports/`: analysis notes and solution writeups

## First Steps

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
bash scripts/download_data.sh
```

Raw data should remain unchanged after download.
