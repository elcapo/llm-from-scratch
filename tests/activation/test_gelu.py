import torch
from scratch.activation.gelu import Gelu

def test_gelu():
    # Prepare
    gelu = Gelu()
    x = torch.linspace(-3, 3, 5)
    # Act
    y = gelu(x)
    # Assert
    assert torch.allclose(
        y,
        torch.tensor([-0.0036, -0.1004,  0.0000,  1.3996,  2.9964]),
        atol=1e-4)
