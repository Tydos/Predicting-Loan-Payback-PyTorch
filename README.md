## Credit Risk Inference System

This project provides a FastAPI-based inference service for credit risk prediction. The service can be containerized and run using Docker.

### Build and Run Docker Image

To build the Docker image, run:

```bash
docker build -f api/inference.dockerfile -t inference .
```

To run the image:

```bash
docker run --env-file .env -p 8000:8000 -d inference
```

Build and run Mlflow 

```bash
docker compose up
```
Why Docker Compose instead of running docker images?
- Allows passing AWS secrets into the container
- Required for connecting to S3 (artifact storage) and PostgreSQL (metadata storage)

### CI/CD on Github Actions
The deploy.yaml workflow automates the deployment pipeline for the API to a hosted EC2 instance. It performs the following steps:

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

### AWS Deployment Notes

1. Create a IAM role with the following permissions

   
| Policy / Role                        | Purpose                                                                 |
| ------------------------------------ | ----------------------------------------------------------------------- |
| `AmazonEC2ContainerRegistryPullOnly` | Pull images from ECR repositories                                       |
| `AmazonEC2ContainerRegistryReadOnly` | Read ECR metadata and repository info                                   |
| `mlflow_s3_access` (custom)          | Access MLFlow production models in S3 (`s3:GetObject`, `s3:ListBucket`) |

### Future Ideas 

1. Setup Canary/Blue-Green deployments of ML models
2. Adding a auto scaling group for horizontal scaling of the servers
3. Adding a load balancer server
