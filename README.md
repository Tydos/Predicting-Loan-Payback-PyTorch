1. Run project: python -m src.main
2. Run fastapi server: uvircorn api.main:app --reload
3. Run MLFlow server(local): mlflow ui
4. Push Docker images to ECR: aws --version, aws configure, aws ecr create-repository --repository-name my-app --region us-east-1, aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin  577124901795.dkr.ecr.us-east-1.amazonaws.com/my-app
5. Connect to EC2 instance: ssh -i "admin.pem" ubuntu@ec2-98-80-179-154.compute-1.amazonaws.com -> install docker and pull ECR images
6. Start MLFlow server(cloud backups): mlflow server --backend-store-uri postgresql://postgresql:<password>!@loan-prediction-metadata.c4jskc8imd2i.us-east-1.rds.amazonaws.com:5432/mlflow

