import pandas as pd

from core.preprocessing import application_to_features, process_data


def test_process_data_fit_transform_round_trip(sample_training_frame):
    feature_frame = sample_training_frame.drop(columns=["loan_paid_back"])

    train_processed, scaler, encoders = process_data(feature_frame, train=True)
    infer_processed, _, _ = process_data(feature_frame, scaler=scaler, encoders=encoders, train=False)

    assert scaler is not None
    assert encoders
    assert list(train_processed.columns) == list(infer_processed.columns)
    assert train_processed.shape == infer_processed.shape


def test_application_to_features_returns_float_vector(
    sample_application, fitted_preprocessing, config
):
    scaler, encoders, _ = fitted_preprocessing

    features = application_to_features(
        sample_application,
        scaler,
        encoders,
        config.dataset.features,
    )

    assert len(features) == len(config.dataset.features)
    assert all(isinstance(value, float) for value in features)


def test_process_data_drops_id_column():
    frame = pd.DataFrame(
        [
            {
                "id": 1,
                "loan_amount": 10_000.0,
                "grade": "B",
            }
        ]
    )

    processed, _, _ = process_data(frame, train=True)

    assert "id" not in processed.columns
