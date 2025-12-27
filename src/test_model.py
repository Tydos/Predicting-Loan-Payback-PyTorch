import torch

# Dummy test function
def dummy_test(model):
    test_input = torch.zeros(1, 11)  # batch of 1, 11 features
    model.eval()
    with torch.no_grad():
        output = model(test_input)
    return torch.isfinite(output).all().item()
