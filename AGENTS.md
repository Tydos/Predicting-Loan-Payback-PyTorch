# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository.

## Project Summary

Credit Risk Scorer trains an XGBoost classifier on LendingClub loan data to predict loan
repayment vs. default. Models are exported to ONNX and registered on DagsHub-hosted MLflow
with a "champion" alias for the production model. A FastAPI service loads the champion model
at startup and serves prediction, health, and results endpoints plus a small scoring UI. The
project deploys to AWS EC2 (Docker) or AWS Lambda (container image).

## Repo Map

```
core/               Framework-agnostic building blocks (config schema, preprocessing,
                     ONNX/metrics helpers, PyTorch architecture, request/response schema)
pipeline/           ETL (raw CSV -> dataset/train.csv) and dataset prep (splits + fitted
                     scaler/encoders), used by both training scripts
services/train/     Training entrypoints: baselines.py (sklearn), train.py (XGBoost, the
                     one that gets promoted to champion), pytorch_train.py (experimental,
                     never promoted), trainer.py (shared PyTorch train/eval loop)
services/inference/ FastAPI app (app.py), static scoring UI, Lambda handler (Mangum)
tests/              pytest suite (unit tests) + load_test.py (Locust, run separately)
notebooks/          Standalone analysis scripts (feature_analysis.py, profile_dataset.py)
docs/               Docs (CHANGELOG.md, DEPLOYMENT.md, EDA.md) + generated artifacts (e.g.
                     train_profile.html)
core/config.yaml    Single source of runtime config, validated by core/config.py (pydantic)
```

## Documentation Map

- `README.md` (root) — project overview and pointers only; kept slim, no run commands.
- `AGENTS.md` (root) — this file: setup, all run commands, conventions.
- `docs/DEPLOYMENT.md` — EC2 and Lambda deployment (secrets, one-time AWS setup, manual deploy
  steps).
- `docs/EDA.md` — feature selection rationale and data findings.
- `docs/CHANGELOG.md` — notable changes, in Keep a Changelog format.

Only `README.md` and `AGENTS.md` live at the repo root — all other markdown docs go in `docs/`.
Keep deployment/infra detail in `docs/DEPLOYMENT.md`, not `README.md` — the README should stay
focused on getting the project running locally.

Data flows: `pipeline/etl.py` → `dataset/train.csv` → `pipeline/prepare.py` (splits + scaler/
encoders) → `services/train/{baselines,train,pytorch_train}.py` (log to MLflow) →
`services/inference/app.py` (loads the `champion` alias at startup).

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync --extra dev          # install runtime + dev (pytest, ruff) deps
cp .env.example .env         # fill in DAGSHUB_USER_TOKEN and ADMIN at minimum
```

`torch` is only needed for the experimental PyTorch path: `uv sync --extra pytorch`. no need to run 'requirement-eda' and no need to update pyproject.toml with the requirement-eda dependencies


## Common Commands

Set your DagsHub token before running anything that touches MLflow: `export DAGSHUB_USER_TOKEN=<your_token>`.

### Test & Lint

```bash
uv run pytest                              # run tests (uses [tool.pytest.ini_options])
uv run ruff check .                        # lint (select = E, F, I, UP; line-length 100)
uv run ruff check --fix .                  # autofix lint issues
```

### Full Pipeline — Docker Compose (recommended)

```bash
docker compose run --rm train              # train + log to DagsHub MLflow
docker compose up inference                # serve the production (champion) model
```

### Full Pipeline — Without Docker

```bash
# 1. ETL — only needed when regenerating dataset/train.csv
PYTHONPATH=. python pipeline/etl.py

# 2. Baselines — majority-class + logistic regression, logged to MLflow
PYTHONPATH=. python services/train/baselines.py

# 3. XGBoost training — trains, exports ONNX, registers + (conditionally) promotes to champion
PYTHONPATH=. python services/train/train.py

# 4. Optional — experimental PyTorch training, never promoted to champion
uv sync --extra pytorch
PYTHONPATH=. python services/train/pytorch_train.py

# 5. Inference API
PYTHONPATH=. uvicorn services.inference.app:app --reload
```

Open [http://localhost:8000](http://localhost:8000) for the built-in scoring UI once the API is
running.

### Data Versioning (DVC)

```bash
dvc pull          # download data
dvc repro         # re-run ETL if config changed
dvc push          # upload new data to remote
```

### Exploratory Analysis

```bash
PYTHONPATH=. python notebooks/feature_analysis.py

# Dataset profiling report (needs the separate .venv-eda — see Setup above)
PYTHONPATH=. .venv-eda/bin/python notebooks/profile_dataset.py   # generates docs/train_profile.html
```

## Conventions

- Always check for the following: bugs, edge cases, PEP8 and docstrings, ruff and linter results, adherence to the DRY principle, and whether logic is broken down into simple steps instead of complex joins or one-line array calculations.

- Python >=3.11, type hints on function signatures, `pydantic` models for config/schema
  validation (see `core/config.py`, `core/schema.py`).
- Formatting/linting is `ruff` with `line-length = 100`; keep imports sorted (`I` rules).
- Config values belong in `core/config.yaml` / `core/config.py`. If you add a config field,
  make sure code actually reads it from the loaded `ValidateConfig` object — don't hardcode
  the same value elsewhere (a few pre-existing fields, e.g. `RawDatasetConfig.raw_target_column`,
  `DatasetConfig.train_path`, `MLflowConfig.model_uri`, are unused for this reason; fixing that
  is on the backlog, not a pattern to copy).
- MLflow experiment/model naming, DagsHub repo owner/name, and the `champion` alias name are
  the integration points between training and inference — don't rename them independently.
- `services/train/train.py::_register_and_promote` only aliases a new model version as
  `champion` if its `val_auc` is >= the current champion's (or there is no champion yet).
  Preserve this gating if you touch promotion logic — don't revert to "always promote".
- Never commit `.env` (it holds real tokens) — it's gitignored; keep `.env.example` as the
  placeholder reference.
- The `/reload` endpoint and other admin actions are protected by the `ADMIN` env var compared
  with `secrets.compare_digest` — keep constant-time comparisons for any new auth checks.

## Testing Notes

- `tests/conftest.py` provides shared fixtures (`config`, `sample_training_frame`,
  `fitted_preprocessing`, `sample_application`).
- `test_architecture.py` is skipped unless `torch` is installed (`pytest.importorskip`).
- `test_inference_app.py` mocks MLflow/DagsHub — it doesn't hit real MLflow/DagsHub.
- `tests/load_test.py` is a Locust script, not part of the pytest suite — it exercises the
  running `/predict` endpoint. Run with:

  ```bash
  locust -f tests/load_test.py --host http://localhost:8000
  ```

## Before You Finish

- Run `uv run pytest` and `uv run ruff check .`; both should pass cleanly.
- Update `docs/CHANGELOG.md` under `[Unreleased]` for any user-facing or behavioral change.
