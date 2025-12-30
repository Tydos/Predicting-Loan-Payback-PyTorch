import mlflow
mlflow.set_experiment("LoanPayback_Experiment")
def register_and_promote(model,run_id,test):
    mlflow.set_tracking_uri("http://localhost:5000")
    
    #Log the model execution 
    mlflow.pytorch.log_model(model, artifact_path="models")

    # Register the model in Model Registry (creates a new version automatically)
    MODEL_NAME = "LoanPayback"
    result = mlflow.register_model(
    f"runs:/{run_id}/models",MODEL_NAME
    )
    print(f"Registered {MODEL_NAME} version {result.version}")

    
    client = mlflow.tracking.MlflowClient()
    latest_versions = client.get_latest_versions(MODEL_NAME, stages=[])
    latest_version_number = max(int(v.version) for v in latest_versions)
    print(f"Latest version for testing: {latest_version_number}")
    model_for_test = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}/{latest_version_number}")

    # Run dummy test
    if test(model_for_test):
        print("Dummy test passed! Promoting to Production...")
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=latest_version_number,
            stage="Production"
        )
        print(f"Model version {latest_version_number} is now in Production.")
    else:
        print("Dummy test failed. Model not promoted.")
