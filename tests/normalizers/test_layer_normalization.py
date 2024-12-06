import torch
from scratch.normalizers.layer_normalization import LayerNorm

def test_layer_normalization():
    # Prepare
    torch.manual_seed(123)
    batch_example = torch.rand(2, 5)
    # Act
    normalization = LayerNorm(emb_dim=5)
    output = normalization(batch_example)
    # Assert
    assert torch.allclose(
        output.mean(dim=-1, keepdim=True),
        torch.tensor([[-0.0], [0.0]]),
        atol = 1e-6)
    assert torch.allclose(
        output.var(dim=-1, keepdim=True, unbiased=False),
        torch.tensor([[1.0], [1.0]]),
        atol = 1e-3)
