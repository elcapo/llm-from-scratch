import torch
from scratch.normalizers.add_one_normalizer import AddOneNormalizer

def test_add_one_normalizer():
    # Prepare
    inputs = torch.tensor([.9544, 1.4950, 1.4754, .8434, .7070, 1.0865])
    # Act
    normalizer = AddOneNormalizer()
    normalized = normalizer(inputs)
    # Assert
    assert torch.allclose(
        normalized,
        torch.tensor([0.1455, 0.2278, 0.2249, 0.1285, 0.1077, 0.1656]),
        atol=1e-4)
