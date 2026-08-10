# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project does not yet follow a formal version scheme beyond the `0.1.0` baseline in
`pyproject.toml`.

## [Unreleased]

### Fixed

- **`dagshub` dependency silently resolved to a 2022-era version (`0.2.7`) lacking
  `dagshub.init()`.** The `eda` optional-dependency group (`ydata-profiling` -> `dacite>=1.9,<2`)
  conflicted with `dagshub`'s `dacite~=1.6.0` pin, and unified extras resolution (`uv lock`)
  downgraded `dagshub` far enough to satisfy both — breaking `dagshub.init()` calls in
  `services/inference/app.py`, `services/train/train.py`, `services/train/baselines.py`, and
  `services/train/pytorch_train.py`. Fixed by pinning `dagshub>=0.7.1` and moving EDA-only
  dependencies out of `pyproject.toml` into `requirements-eda.txt`, installed in the
  pre-existing separate `.venv-eda` environment.
- **Champion model promotion never compared against the current champion.**
  `services/train/train.py::_register_and_promote` promoted every newly trained model to the
  `champion` alias as long as a trivial sanity check passed (a finite prediction on a
  zero-vector input), regardless of whether it performed worse than the existing champion.
  Added `_get_champion_val_auc()` and gated promotion on the new model's `val_auc` being `>=`
  the current champion's (or there being no champion yet).

### Added

- `AGENTS.md` (root) — guidance for AI coding agents working in this repo (setup, commands,
  conventions, known gotchas).
- `docs/CHANGELOG.md` (this file).
- `docs/DEPLOYMENT.md` — EC2 and Lambda deployment steps, secrets, and one-time AWS setup, moved
  out of `README.md` to keep the README focused on getting the project running locally.
- `requirements-eda.txt` for the standalone EDA/profiling environment (`.venv-eda`).

### Changed

- `README.md`: replaced the "EC2 Deployment" and "Lambda Deployment" sections with a short
  summary and a link to `docs/DEPLOYMENT.md`; added a "Docs" line linking to `docs/EDA.md`,
  `docs/DEPLOYMENT.md`, `docs/CHANGELOG.md`, and `AGENTS.md`.
- Moved `EDA.md`, `DEPLOYMENT.md`, and `CHANGELOG.md` from the repo root into `docs/`, so only
  `README.md` and `AGENTS.md` remain at the root.
- Moved all run commands (setup, ETL, DVC, training, inference, EDA, load testing) out of
  `README.md` into `AGENTS.md`'s "Common Commands" section. `README.md` now stays prose-only
  with pointers, so agents/devs have one authoritative place for exact commands.

### Known Issues (tracked, not yet fixed)

- `core/config.py` declares `RawDatasetConfig.raw_target_column`, `DatasetConfig.train_path`,
  and `MLflowConfig.model_uri`; all three are set in `core/config.yaml` but never read from the
  loaded config object anywhere in the codebase (the same values are hardcoded at the call
  sites instead). Either wire them up or remove them.
- `tests/conftest.py` imports `OrdinalEncoder` and `StandardScaler` from `sklearn.preprocessing`
  without using them.
- `pipeline/etl.py::handle_missing_values` will raise `IndexError` if a categorical column is
  entirely `NaN` (`.mode()` returns an empty frame in that case).
- EC2 and Lambda deployments push to the same ECR tag (`credit-models:latest`) — redeploying
  one overwrites the image the other expects (documented in `docs/DEPLOYMENT.md`).

## [0.1.0] - Baseline

Initial state of the project prior to this changelog's introduction. Established:

- ETL pipeline (`pipeline/etl.py`) turning the raw LendingClub CSV into a cleaned
  `dataset/train.csv`, versioned with DVC.
- Dataset prep (`pipeline/prepare.py`) producing stratified train/val/test splits with a
  fitted `StandardScaler` and per-column `OrdinalEncoder`s.
- Baseline models (majority-class, logistic regression) and an XGBoost classifier, exported to
  ONNX and logged to DagsHub-hosted MLflow with a `champion` alias for production.
- Experimental PyTorch training path (`services/train/pytorch_train.py`), not promoted to
  champion.
- FastAPI inference service (`services/inference/app.py`) with `/predict`, `/health_check`,
  `/schema`, `/results`, and an admin-gated `/reload` endpoint, plus a built-in scoring UI.
- Docker Compose for local training/serving; CI/CD workflows for unit tests, ETL, EC2
  deployment, and Lambda deployment.
- Unit test suite covering config loading, schema validation, preprocessing, ETL, ONNX
  round-tripping, and the inference API.
