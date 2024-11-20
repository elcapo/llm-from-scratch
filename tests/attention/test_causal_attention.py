from torch import manual_seed, stack, tensor, Size
from scratch.attention.causal_attention import CausalAttention

def test_causal_attention_compute():
    # Prepare
    manual_seed(123)
    inputs = tensor([
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55]])
    batch = stack((inputs, inputs), dim=0)
    attention = CausalAttention(d_in=3, d_out=2, context_length=batch.shape[1])
    # Act
    context_vectors = attention(batch)
    # Assert
    assert context_vectors.shape == Size([2, 6, 2])
