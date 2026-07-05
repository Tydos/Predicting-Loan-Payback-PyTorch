from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def inference_client(fitted_preprocessing, monkeypatch):
    scaler, encoders, _ = fitted_preprocessing
    session = MagicMock()

    mlflow_client = MagicMock()
    mlflow_client.search_experiments.return_value = []
    mlflow_client.get_experiment_by_name.return_value = None

    import services.inference.app as app_module

    monkeypatch.setattr(app_module.dagshub, "init", MagicMock())
    monkeypatch.setattr(
        app_module,
        "_load_model",
        lambda client, model_name: (session, 7, "models:/LoanPayback@champion"),
    )
    monkeypatch.setattr(
        app_module,
        "_load_preprocessing",
        lambda client, model_name, version: (scaler, encoders),
    )
    monkeypatch.setattr(app_module, "MlflowClient", lambda: mlflow_client)
    monkeypatch.setattr(app_module, "predict_proba", lambda sess, features: 0.75)

    with TestClient(app_module.app) as client:
        yield client


def test_health_check_reports_ready_model(inference_client):
    response = inference_client.get("/health_check")

    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is True
    assert body["preprocessing_available"] is True
    assert body["model_version"] == 7
    assert body["message"] == "ready"
    assert body["mlflow_reachable"] is True


def test_schema_endpoint_exposes_feature_metadata(inference_client):
    response = inference_client.get("/schema")

    assert response.status_code == 200
    body = response.json()
    assert len(body["features"]) == 10
    assert len(body["fields"]) == 10
    assert "loan_amount" in body["fields"]
    assert body["fields"]["credit_score"]["type"] == "integer"
    assert body["fields"]["grade"]["options"] == ["A", "B", "C", "D", "E", "F", "G"]
    assert body["sample_application"]["loan_amount"] == 12_000.0
    assert body["preprocessing_available"] is True
    assert body["target"] == "loan_paid_back"


def test_predict_returns_binary_prediction(inference_client, sample_application):
    response = inference_client.post("/predict", json=sample_application)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert body["prediction_label"] in ("paid_back", "default")
    assert 0 <= body["paid_back_probability"] <= 1
    assert "inference_latency_ms" in body
    assert body["model_version"] == 7


def test_reload_requires_admin_key(inference_client, monkeypatch):
    monkeypatch.setenv("ADMIN", "test-admin-key")

    unauthorized = inference_client.get("/reload", headers={"X-API-Key": "wrong-key"})
    authorized = inference_client.get("/reload", headers={"X-API-Key": "test-admin-key"})

    assert unauthorized.status_code == 403
    assert authorized.status_code == 200
    assert authorized.json()["status"] == "reloaded"
