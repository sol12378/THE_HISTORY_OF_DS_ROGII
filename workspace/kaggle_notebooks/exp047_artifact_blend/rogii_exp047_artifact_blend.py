from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

# Keep the primary blend conservative. A diagnostic 70/30 blend is also saved.
SUNNY_WEIGHT = 0.80
WORKING_DIR = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()


# ===== CELL =====

# Locate helper dataset and import the orchestration utilities.

def find_helper_root() -> Path:
    candidates = [
        Path('/kaggle/input/rogii-sunny-v10-stack-helpers/rogii_sunny_v10_stack_helpers'),
        Path('/kaggle/input/rogii-readable-sunny-v10-stack-helpers/rogii_sunny_v10_stack_helpers'),
        Path('/kaggle/input/rogii_sunny_v10_stack_helpers'),
        Path.cwd() / 'rogii_sunny_v10_stack_helpers',
    ]
    for path in candidates:
        if (path / 'blend_components.py').is_file():
            return path
    input_root = Path('/kaggle/input')
    if input_root.exists():
        for file in input_root.glob('**/blend_components.py'):
            root = file.parent
            if (root / 'sunny_physical_component.py').is_file() and (root / 'v10_artifact_component.py').is_file():
                return root
    raise FileNotFoundError('Helper dataset not found. Attach kojimar/rogii-sunny-v10-stack-helpers.')

HELPER_ROOT = find_helper_root()
sys.path.insert(0, str(HELPER_ROOT))
print('HELPER_ROOT =', HELPER_ROOT)

from blend_components import (  # noqa: E402
    ComponentPaths,
    component_disagreement,
    find_competition_dir,
    read_submission,
    run_python_component,
    sha256_file,
    summarize_submission,
    weighted_blend,
)

COMPETITION_DIR = find_competition_dir()
SAMPLE_PATH = COMPETITION_DIR / 'sample_submission.csv'
print('COMPETITION_DIR =', COMPETITION_DIR)
print('SAMPLE_PATH =', SAMPLE_PATH)


# ===== CELL =====

paths = ComponentPaths(
    helper_root=HELPER_ROOT,
    working_dir=WORKING_DIR,
    sample_submission=SAMPLE_PATH,
)

sample = pd.read_csv(paths.sample_submission)[['id']]
print('sample rows:', len(sample))
print(sample.head().to_string(index=False))


# ===== CELL =====

sunny_csv = run_python_component(
    name='sunny_physical',
    script_path=paths.sunny_script,
    output_dir=WORKING_DIR / 'component_sunny_physical',
)

sunny = read_submission(sunny_csv, sample, 'sunny_physical')
summarize_submission(sunny, 'sunny_physical', sunny_csv)


# ===== CELL =====

v10_output_dir = WORKING_DIR / 'component_v10_artifact_stack'

v10_csv = run_python_component(
    name='v10_artifact_stack',
    script_path=paths.v10_script,
    output_dir=v10_output_dir,
    extra_env={
        'ROGII_INFERENCE_ONLY': '1',
        'ROGII_SAVE_ARTIFACTS': '0',
        'ROGII_RUN_TABICL': '1',
        'ROGII_OUTPUT_DIR': str(v10_output_dir),
    },
)

v10 = read_submission(v10_csv, sample, 'v10_artifact_stack')
summarize_submission(v10, 'v10_artifact_stack', v10_csv)


# ===== CELL =====

submission = weighted_blend(
    sample=sample,
    primary=sunny,
    secondary=v10,
    primary_weight=SUNNY_WEIGHT,
)

submission_path = WORKING_DIR / 'submission.csv'
submission.to_csv(submission_path, index=False)

# Diagnostic only. This is not the official submission file.
diagnostic_70_30 = weighted_blend(sample, sunny, v10, primary_weight=0.70)
diagnostic_70_30_path = WORKING_DIR / 'diagnostic_submission_sunny70_v10_30.csv'
diagnostic_70_30.to_csv(diagnostic_70_30_path, index=False)

disagreement = component_disagreement(sunny, v10)
summary = {
    'sunny_weight': SUNNY_WEIGHT,
    'v10_weight': 1.0 - SUNNY_WEIGHT,
    'submission_path': str(submission_path),
    'submission_sha256': sha256_file(submission_path),
    'diagnostic_70_30_path': str(diagnostic_70_30_path),
    'rows': int(len(submission)),
    'null_count': int(submission['tvt'].isnull().sum()),
    **disagreement,
}

print(json.dumps(summary, indent=2))
print(submission.head(8).to_string(index=False))
