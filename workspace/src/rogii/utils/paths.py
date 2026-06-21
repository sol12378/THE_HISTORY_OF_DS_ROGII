from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
FOLDS = DATA / "folds"
EXPERIMENTS = ROOT / "experiments"
OUTPUTS = ROOT / "outputs"
