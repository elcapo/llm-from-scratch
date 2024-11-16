from torch import allclose, tensor
from scratch.normalizers.add_one_normalization import AddOneNormalization

def test_add_one_normalization():
    # Prepare
    input = tensor([.9544, 1.4950, 1.4754, .8434, .7070, 1.0865])
    # Act
    normalization = AddOneNormalization()
    normalized = normalization.normalize(input)
    # Assert
    assert allclose(
        normalized,
        tensor([0.1455, 0.2278, 0.2249, 0.1285, 0.1077, 0.1656]),
        atol=1e-04
    )
