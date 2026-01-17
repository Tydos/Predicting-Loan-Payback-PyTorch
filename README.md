## Credit Risk Inference System

This project provides a FastAPI-based inference service for credit risk prediction. The service can be containerized and run using Docker.

### Build and Run Docker Image

To build the Docker image, run:

```bash
docker build -f api/inference.dockerfile -t inference .
```

To run the image:

```bash
docker run -p 8000:8000 inference
```

Build and run Mlflow 

```bash
docker compose up
```
Why Docker Compose instead of running docker images?
- Allows passing AWS secrets into the container
- Required for connecting to S3 (artifact storage) and PostgreSQL (metadata storage)

### CI/CD on Github Actions
The deploy.yaml workflow automates the deployment pipeline for the API. It performs the following steps:

1. AWS Authorization
- Authorizes the GitHub Runner to access AWS
- Required for pushing Docker images to ECS and deploying resources

2. Build & Push Docker Image
- Builds the API Docker image
- Pushes the image to Elastic Container Service (ECS)

3. Deploy to EC2
- Connects to the EC2 instance hosting the API
- Boots the new API code, replacing the old version

This CI/CD setup ensures automated, secure, and consistent deployment of API updates.

<!--
1. Run project: python -m src.main
2. Run fastapi server: uvircorn api.main:app --reload
3. Run MLFlow server(local): mlflow ui
4. Push Docker images to ECR: aws --version, aws configure, aws ecr create-repository --repository-name my-app --region us-east-1, aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin  577124901795.dkr.ecr.us-east-1.amazonaws.com/my-app
5. Connect to EC2 instance: ssh -i "admin.pem" ubuntu@ec2-98-80-179-154.compute-1.amazonaws.com -> install docker and pull ECR images
6. Start MLFlow server(cloud backups): mlflow server --backend-store-uri postgresql://postgresql:<password>!@loan-prediction-metadata.c4jskc8imd2i.us-east-1.rds.amazonaws.com:5432/mlflow
 -->
