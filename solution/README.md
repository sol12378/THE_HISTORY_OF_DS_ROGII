# ROGII Solution Entrypoints

This folder contains thin wrappers that connect the inherited ROGII workspace to the shared `autoresearch` controls.

Use `workspace/scripts/` for historical experiment code. Use the shared CLI for submission checks and ledger registration:

```powershell
.\.venv\Scripts\python.exe -m autoresearch.cli --config competitions\rogii-wellbore-geology-prediction\config.yaml check-submission <submission.csv>
.\.venv\Scripts\python.exe -m autoresearch.cli --config competitions\rogii-wellbore-geology-prediction\config.yaml submit <submission.csv> --note "rogii experiment note"
```
