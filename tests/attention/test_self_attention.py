from torch import allclose, manual_seed, tensor
from scratch.attention.self_attention import SelfAttention

def test_self_attention_compute():
    # Prepare
    manual_seed(123)
    attention = SelfAttention(d_in=3, d_out=2)
    inputs = tensor([
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55]])
    # Act
    context_vectors = attention(inputs)
    # Assert
    assert allclose(
        context_vectors[1],
        tensor([0.3061, 0.8210]),
        atol=1e-4)