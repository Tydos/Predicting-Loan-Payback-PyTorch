import pytest


def test_forward_output_shape():
    torch = pytest.importorskip("torch")
    from core.architecture import LoanPredictor

    model = LoanPredictor(num_features=10, hidden_layers=[8, 4], dropout=0.1)
    inputs = torch.randn(3, 10)

    outputs = model(inputs)

    assert outputs.shape == (3, 1)
