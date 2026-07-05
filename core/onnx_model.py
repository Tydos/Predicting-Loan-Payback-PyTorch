import numpy as np
import onnxruntime as ort
from onnx import ModelProto
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from xgboost import XGBClassifier


def export_xgboost(model: XGBClassifier, n_features: int) -> ModelProto:
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    return convert_xgboost(model.get_booster(), initial_types=initial_type)


def create_session(model_proto: ModelProto) -> ort.InferenceSession:
    return ort.InferenceSession(model_proto.SerializeToString())


def predict_proba(session: ort.InferenceSession, features: list[float]) -> float:
    input_name = session.get_inputs()[0].name
    x = np.array([features], dtype=np.float32)
    outputs = session.run(None, {input_name: x})
    output_names = [o.name for o in session.get_outputs()]

    for name, values in zip(output_names, outputs, strict=True):
        if name == "probabilities" or (len(values.shape) == 2 and values.shape[1] == 2):
            return float(values[0, 1])

    raise ValueError("Could not find probability output in ONNX model")
