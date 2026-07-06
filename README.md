## Credit Risk Scorer

Credit Risk Scorer trains an XGBoost classifier on LendingClub loan data to predict repayment vs default. Models export to ONNX and register on DagsHub MLflow with experiment tracking, artifact storage, and champion promotion. A FastAPI inference service loads the champion model at startup, serves a scoring UI, and exposes prediction, health, and results endpoints. The pipeline runs via Docker Compose locally or deploys to AWS Lambda as a container image.

## What it does

<p align="center" width="100%">
<video src="https://github-production-user-asset-6210df.s3.amazonaws.com/80669588/614281587-ba20f8aa-8fd9-4994-8236-60a81b0efde3.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20260628%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260628T183104Z&X-Amz-Expires=300&X-Amz-Signature=3d9be4fa72eeee8cb236592b195cc65fbbf12fcf4d08d7dd11f683bb3b74b03b&X-Amz-SignedHeaders=host&response-content-type=video%2Fmp4" width="80%" controls></video>
</p>

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
---

## EC2 Deployment

The inference API can run on a persistent EC2 instance as a Docker container (uvicorn via [`services/inference/inference.dockerfile`](services/inference/inference.dockerfile)). Deployments are triggered manually through [`.github/workflows/deploy.yaml`](.github/workflows/deploy.yaml) (`workflow_dispatch` only — auto-deploy on push is disabled).

The workflow builds the inference image, pushes it to private ECR, SSHes into the instance, pulls the image, and restarts the `loanpayback` container on port 80.

**GitHub secrets:**

| Secret | Example |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM access key for CI |
| `AWS_SECRET_ACCESS_KEY` | IAM secret key |
| `AWS_REGION` | `us-east-1` |
| `ECR_REGISTRY` | `031749757344.dkr.ecr.us-east-1.amazonaws.com` |
| `ECR_REPOSITORY` | `credit-models` |
| `EC2_HOST` | Public IP or DNS of the instance |
| `EC2_USER` | `ec2-user` (Amazon Linux) or `ubuntu` |
| `EC2_SSH_KEY` | Private key contents for SSH |

**One-time EC2 setup:**

```bash
# On the instance: install Docker + AWS CLI (Amazon Linux 2023 example)
sudo dnf install -y docker awscli
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user

# Create ECR repo (if not already created)
aws ecr create-repository --repository-name credit-models

# Security group: allow inbound TCP 80 (and 22 for SSH)
```

Store `DAGSHUB_USER_TOKEN` and `ADMIN` on the instance (e.g. in `~/.env`) — the container needs them at runtime to reach MLflow and protect `/reload`.

**Manual deploy** (from repo root):

```bash
source .venv/bin/activate
set -a && source .env && set +a

aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin 031749757344.dkr.ecr.us-east-1.amazonaws.com

docker build -f services/inference/inference.dockerfile -t credit-models .
docker tag credit-models:latest \
  031749757344.dkr.ecr.us-east-1.amazonaws.com/credit-models:latest
docker push 031749757344.dkr.ecr.us-east-1.amazonaws.com/credit-models:latest

# On EC2 (or via SSH)
docker stop loanpayback || true && docker rm -f loanpayback || true
docker pull 031749757344.dkr.ecr.us-east-1.amazonaws.com/credit-models:latest
docker run -d \
  -p 80:8000 \
  --name loanpayback \
  --env-file ~/.env \
  031749757344.dkr.ecr.us-east-1.amazonaws.com/credit-models:latest
```

The API is then available at `http://<EC2_HOST>/`. Unlike Lambda, the container stays warm — no cold starts after the initial model load.

> **Note:** EC2 and Lambda currently push to the same ECR tag (`credit-models:latest`) but use different Dockerfiles. Redeploying one overwrites the image the other expects — use separate tags (e.g. `:ec2` and `:lambda`) if running both in production.

---

---

## Lambda Deployment

The inference API runs on AWS Lambda as a container image. Pushes to `main` trigger [`.github/workflows/deploy-lambda.yaml`](.github/workflows/deploy-lambda.yaml), which builds and pushes to **private ECR** (Lambda does not support ECR Public for function images).

**GitHub secrets** (in addition to existing AWS / EC2 creds):

| Secret | Example |
|---|---|
| `LAMBDA_FUNCTION_NAME` | `credit-risk-inference` |
| `ECR_REGISTRY` | `031749757344.dkr.ecr.us-east-1.amazonaws.com` |
| `ECR_REPOSITORY` | `credit-models` |

**Manual deploy** (from repo root):

```bash
source .venv/bin/activate
set -a && source .env && set +a

aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin 031749757344.dkr.ecr.us-east-1.amazonaws.com

docker build --provenance=false --sbom=false \
  -f services/inference/lambda.dockerfile -t credit-models .

docker tag credit-models:latest \
  031749757344.dkr.ecr.us-east-1.amazonaws.com/credit-models:latest

docker push 031749757344.dkr.ecr.us-east-1.amazonaws.com/credit-models:latest

aws lambda update-function-code \
  --function-name credit-risk-inference \
  --image-uri 031749757344.dkr.ecr.us-east-1.amazonaws.com/credit-models:latest

aws lambda wait function-updated-v2 --function-name credit-risk-inference
```

**One-time AWS setup:**

```bash
# IAM role for Lambda
aws iam create-role \
  --role-name lambda-inference-role \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "lambda.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

aws iam attach-role-policy \
  --role-name lambda-inference-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam attach-role-policy \
  --role-name lambda-inference-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly

# Private ECR repo (if not already created)
aws ecr create-repository --repository-name credit-models

# Create Lambda function (secrets via env, not baked into image)
aws lambda create-function \
  --function-name credit-risk-inference \
  --package-type Image \
  --code ImageUri=031749757344.dkr.ecr.us-east-1.amazonaws.com/credit-models:latest \
  --role arn:aws:iam::ACCOUNT_ID:role/lambda-inference-role \
  --timeout 120 \
  --memory-size 3008 \
  --architectures x86_64 \
  --environment "Variables={
    DAGSHUB_REPO_OWNER=pjawale,
    DAGSHUB_REPO_NAME=credit-scorer,
    DAGSHUB_USER_TOKEN=$DAGSHUB_USER_TOKEN,
    ADMIN=$ADMIN
  }"

# Public Function URL
aws lambda create-function-url-config \
  --function-name credit-risk-inference \
  --auth-type NONE \
  --cors '{"AllowOrigins":["*"],"AllowMethods":["*"],"AllowHeaders":["*"]}'

aws lambda add-permission \
  --function-name credit-risk-inference \
  --statement-id FunctionURLAllowPublicAccess \
  --action lambda:InvokeFunctionUrl \
  --principal "*" \
  --function-url-auth-type NONE

aws lambda add-permission \
  --function-name credit-risk-inference \
  --statement-id PublicInvokeFunction \
  --action lambda:InvokeFunction \
  --principal "*"
```

Expect **30–70s cold starts** while the container imports dependencies and loads the ONNX model from MLflow. Warm `/predict` calls are sub-second.
