from torch import allclose, manual_seed, stack, tensor
from scratch.attention.multihead_attention import MultiHeadAttention

def test_multihead_attention():
    # Prepare
    manual_seed(123)
    inputs = tensor([
        [0.43, 0.15, 0.89], # Your     (x^1)
        [0.55, 0.87, 0.66], # journey  (x^2)
        [0.57, 0.85, 0.64], # starts   (x^3)
        [0.22, 0.58, 0.33], # with     (x^4)
        [0.77, 0.25, 0.10], # one      (x^5)
        [0.05, 0.80, 0.55]] # step     (x^6)
    )
    batch = stack((inputs, inputs), dim=0)
    batch_size, context_length, d_in = batch.shape
    d_out = 2
    # Act
    attention = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vectors = attention(batch)
    # Assert
    assert allclose(
        context_vectors,
        tensor([[[0.3190, 0.4858],
                [0.2943, 0.3897],
                [0.2856, 0.3593],
                [0.2693, 0.3873],
                [0.2639, 0.3928],
                [0.2575, 0.4028]],

                [[0.3190, 0.4858],
                [0.2943, 0.3897],
                [0.2856, 0.3593],
                [0.2693, 0.3873],
                [0.2639, 0.3928],
                [0.2575, 0.4028]]]),
        atol=1e-4)