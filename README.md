## Credit Risk Scorer

Credit Risk Scorer trains an XGBoost classifier on LendingClub loan data to predict repayment vs default. Models export to ONNX and register on DagsHub MLflow with experiment tracking, artifact storage, and champion promotion. A FastAPI inference service loads the champion model at startup, serves a scoring UI, and exposes prediction, health, and results endpoints. The pipeline runs via Docker Compose locally or deploys to AWS Lambda as a container image.

**Docs:** [`docs/EDA.md`](docs/EDA.md) (feature selection & data findings) · [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) (EC2 & Lambda) · [`docs/CHANGELOG.md`](docs/CHANGELOG.md) · [`AGENTS.md`](AGENTS.md) (setup, run commands, conventions)

## What it does

<p align="center" width="100%">
<video src="https://github-production-user-asset-6210df.s3.amazonaws.com/80669588/614281587-ba20f8aa-8fd9-4994-8236-60a81b0efde3.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20260628%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260628T183104Z&X-Amz-Expires=300&X-Amz-Signature=3d9be4fa72eeee8cb236592b195cc65fbbf12fcf4d08d7dd11f683bb3b74b03b&X-Amz-SignedHeaders=host&response-content-type=video%2Fmp4" width="80%" controls></video>
</p>

---

## Dataset

The raw data is the **LendingClub Loan Data** publicly available on Kaggle:

> [LendingClub Accepted & Rejected Loans 2007–2018 Q4](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

Download both files into `dataset/`. The raw CSV is turned into a cleaned `dataset/train.csv`
via the ETL pipeline (`pipeline/etl.py`) and tracked with DVC — see [`AGENTS.md`](AGENTS.md) for
the exact commands.

---

## How to Run

The project can run fully via Docker Compose, or step-by-step locally (ETL → baselines →
XGBoost training → inference API). Both paths, plus the optional experimental PyTorch training
run, are documented in [`AGENTS.md`](AGENTS.md#common-commands).

Once the inference API is running, open [http://localhost:8000](http://localhost:8000) for the
built-in scoring UI.

---

## Exploratory Analysis

Feature selection rationale and data findings are written up in [`docs/EDA.md`](docs/EDA.md).
Commands to reproduce the feature analysis script and the `ydata-profiling` report (which needs
its own virtual environment — see [`AGENTS.md`](AGENTS.md#setup)) are in
[`AGENTS.md`](AGENTS.md#common-commands).

---

## Load Testing

A Locust script (`tests/load_test.py`) exercises the `/predict` endpoint — see
[`AGENTS.md`](AGENTS.md#testing-notes) for how to run it.

---

## Deployment

The inference API deploys to either a persistent EC2 instance (Docker container, no cold
starts) or AWS Lambda (container image, pay-per-request with cold starts). Both are driven by
GitHub Actions workflows and use the same FastAPI app with different Dockerfiles.

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for GitHub secrets, one-time AWS setup, and manual
deploy steps for both targets.
