from core.metrics import xy
from core.onnx_model import create_session, export_xgboost, predict_proba
from core.preprocessing import process_data
from xgboost import XGBClassifier


def test_export_and_predict_roundtrip(sample_training_frame):
    target = "loan_paid_back"
    feature_frame = sample_training_frame.drop(columns=[target])
    processed, _, _ = process_data(feature_frame, train=True)
    train_frame = processed.copy()
    train_frame[target] = sample_training_frame[target].values

    x_train, y_train = xy(train_frame, target)

    model = XGBClassifier(n_estimators=2, max_depth=2)
    model.fit(x_train, y_train)

    n_features = x_train.shape[1]
    onnx_model = export_xgboost(model, n_features)
    session = create_session(onnx_model)

    features = x_train[0].tolist()
    prob = predict_proba(session, features)

    assert 0.0 <= prob <= 1.0
