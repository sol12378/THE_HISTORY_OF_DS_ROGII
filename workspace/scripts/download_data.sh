#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
KAGGLE_BIN="${KAGGLE_BIN:-$ROOT_DIR/.venv/bin/kaggle}"

DEFAULT_KAGGLE_DIR="$HOME/.kaggle"

if [ -f "$DEFAULT_KAGGLE_DIR/kaggle.json" ]; then
  export KAGGLE_CONFIG_DIR="$DEFAULT_KAGGLE_DIR"
else
  export KAGGLE_CONFIG_DIR="$ROOT_DIR/.kaggle"
  mkdir -p "$KAGGLE_CONFIG_DIR"

  "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

from dotenv import dotenv_values

env = dotenv_values(".env")
username = (
    os.environ.get("KAGGLE_USERNAME")
    or env.get("KAGGLE_USERNAME")
    or env.get("KAGGLE_API_USERNAME")
)
key = os.environ.get("KAGGLE_KEY") or env.get("KAGGLE_KEY") or env.get("KAGGLE_API_KEY")

if not username or not key:
    raise SystemExit("KAGGLE_USERNAME and KAGGLE_KEY are required. Add them to .env.")

config_dir = Path(os.environ["KAGGLE_CONFIG_DIR"])
config_dir.mkdir(parents=True, exist_ok=True)
config_path = config_dir / "kaggle.json"
config_path.write_text(
    json.dumps({"username": username, "key": key}),
    encoding="utf-8",
)
config_path.chmod(0o600)
PY
fi

if ! "$KAGGLE_BIN" competitions download -c rogii-wellbore-geology-prediction -p data/raw; then
  cat >&2 <<'EOF'

Kaggle download failed.

If the error is 401 Unauthorized, check both:
- .env contains the exact Kaggle username from your downloaded kaggle.json
- .env contains the exact API key from your downloaded kaggle.json
- Competition rules are accepted at:
  https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/rules

Expected .env:
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_API_KEY=your_kaggle_api_key
EOF
  exit 1
fi
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
from zipfile import ZipFile

raw_dir = Path("data/raw")
for zip_path in raw_dir.glob("*.zip"):
    with ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
PY

echo "Downloaded competition files to data/raw"
