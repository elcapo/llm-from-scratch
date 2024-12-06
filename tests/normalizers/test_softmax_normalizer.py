import torch
from scratch.normalizers.softmax_normalizer import SoftmaxNormalizer

def test_softmax_normalizer():
    # Prepare
    inputs = torch.tensor([.9544, 1.4950, 1.4754, .8434, .7070, 1.0865])
    # Act
    normalizer = SoftmaxNormalizer()
    normalized = normalizer(inputs)
    # Assert
    assert torch.allclose(
        normalized,
        torch.tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581]),
        atol=1e-4)
