# Simple Random Forest Project

A minimal project that trains a Random Forest classifier using scikit-learn built-in datasets.

Structure

- `data/` — placeholder for datasets (not required for built-in datasets)
- `models/` — saved model artifacts
- `src/` — source code (training, model, utils)

Quick start (Windows PowerShell)

```powershell
cd GenerativeAI\simple_rf_project
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.train --dataset iris --output models/model.joblib --n-estimators 100
```

Notes

- Supported datasets: `iris`, `synthetic` (default `iris`).
- Model saved to `models/model.joblib` by default.
