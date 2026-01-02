import mlflow.pytorch

def register_and_promote(model, run_id, test_fn):
    MODEL_NAME = "LoanPayback"
    mlflow.pytorch.log_model(model, artifact_path="models")
    result = mlflow.register_model(f"runs:/{run_id}/models", MODEL_NAME)
    print(f"Registered {MODEL_NAME} version {result.version}")

    client = mlflow.MlflowClient()
    latest_version = max(int(v.version) for v in client.get_latest_versions(MODEL_NAME))
    print(f"Latest version for testing: {latest_version}")

    model_for_test = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}/{latest_version}")

    if test_fn(model_for_test):
        print("Dummy test passed! Promoting to Production...")
        client.transition_model_version_stage(MODEL_NAME, latest_version, stage="Production")
        print(f"Model version {latest_version} is now in Production.")
    else:
        print("Dummy test failed. Model not promoted.")
