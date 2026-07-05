## Credit Risk Scorer

Credit Risk Scorer is a containerized MLOps pipeline that trains an untuned XGBoost classifier on ~594,000 historical loan records (sourced from Kaggle) to predict whether a borrower will repay or default. The model is exported to ONNX and tracked remotely on DagsHub's hosted MLflow, which handles experiment logging, artifact storage (scaler, encoders, ONNX model), model registration, and the champion alias that gates production promotion. The inference service is a FastAPI app that loads the champion ONNX model via ONNX Runtime at startup, serves a built-in UI for manual loan application scoring, and exposes endpoints for prediction, model validation results, and health checking — all packaged as Docker images orchestrated via Docker Compose.

## What it does

<p align="center" width="100%">
<video src="https://github-production-user-asset-6210df.s3.amazonaws.com/80669588/614281587-ba20f8aa-8fd9-4994-8236-60a81b0efde3.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20260628%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260628T183104Z&X-Amz-Expires=300&X-Amz-Signature=3d9be4fa72eeee8cb236592b195cc65fbbf12fcf4d08d7dd11f683bb3b74b03b&X-Amz-SignedHeaders=host&response-content-type=video%2Fmp4" width="80%" controls></video>
</p>

---

## Project Structure

```
Credit-Risk-Scorer/
├── core/                        # Shared library used by both services
│   ├── config.py                # Pydantic config models + loader
│   ├── config.yaml              # Single source of truth for all settings
│   ├── architecture.py          # LoanPredictor nn.Module (experimental PyTorch)
│   ├── metrics.py               # Shared validation metrics helpers
│   ├── onnx_model.py            # ONNX export + ONNX Runtime inference helpers
│   ├── preprocessing.py         # Scaler/encoder fit+transform utilities
│   └── schema.py                # LoanApplicationPayload (API request schema)
│
├── pipeline/                    # Data pipeline, in order
│   ├── etl.py                   # Step 1: raw CSV → dataset/train.csv
│   └── prepare.py               # Step 2: train.csv → train/val/test splits
│
├── services/
│   ├── train/
│   │   ├── baselines.py         # Runnable: majority-class + logistic regression
│   │   ├── pytorch_train.py     # Optional: experimental PyTorch training
│   │   ├── trainer.py           # PyTorch training loop (used by pytorch_train.py)
│   │   └── train.py             # Runnable: XGBoost training + ONNX promotion
│   └── inference/
│       └── app.py               # FastAPI inference service
│
├── tests/
│   └── load_test.py             # Locust load test for the inference API
│
├── notebooks/                   # Exploratory analysis scripts
│   ├── feature_analysis.py
│   └── profile_dataset.py
│
└── dataset/                     # DVC-managed data files
```

---

## Dataset

The raw data is the **LendingClub Loan Data** publicly available on Kaggle:

> [LendingClub Accepted & Rejected Loans 2007–2018 Q4](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

Download both files and place them in `dataset/`, then run the ETL to produce the cleaned training file:

```bash
PYTHONPATH=. python pipeline/etl.py
```

---

## Data Versioning

```bash
dvc pull          # download data
dvc repro         # re-run ETL if config changed
dvc push          # upload new data to remote
```

---

## How to Run

Set your DagsHub token before running anything:

```bash
export DAGSHUB_USER_TOKEN=<your_token>
```

### With Docker (recommended)

```bash
# Train (logs to DagsHub MLflow)
docker compose run --rm train

# Serve the production model
docker compose up inference
```

### Without Docker

**Step 1 — ETL** (only needed when regenerating `dataset/train.csv`):

```bash
PYTHONPATH=. python pipeline/etl.py
```

**Step 2 — Baselines** (majority-class + logistic regression, logged to MLflow):

```bash
PYTHONPATH=. python services/train/baselines.py
```

**Step 3 — XGBoost training** (trains default XGBoost, exports ONNX, registers + promotes the model):

```bash
PYTHONPATH=. python services/train/train.py
```

**Optional — PyTorch training** (experimental, not promoted to champion):

```bash
pip install -e ".[pytorch]"
PYTHONPATH=. python services/train/pytorch_train.py
```

**Step 4 — Inference API**:

```bash
PYTHONPATH=. uvicorn services.inference.app:app --reload
```

Then open [http://localhost:8000](http://localhost:8000) for the built-in scoring UI.

---

## Exploratory Analysis

```bash
PYTHONPATH=. python notebooks/feature_analysis.py
PYTHONPATH=. python notebooks/profile_dataset.py   # generates docs/train_profile.html
```

---

## Load Testing

```bash
locust -f tests/load_test.py --host http://localhost:8000
```
