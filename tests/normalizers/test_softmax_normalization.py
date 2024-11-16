from torch import allclose, tensor
from scratch.normalizers.softmax_normalization import SoftmaxNormalization

def test_softmax_normalization():
    # Prepare
    input = tensor([.9544, 1.4950, 1.4754, .8434, .7070, 1.0865])
    # Act
    normalization = SoftmaxNormalization()
    normalized = normalization.normalize(input)
    # Assert
    assert allclose(
        normalized,
        tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581]),
        atol=1e-04
    )
