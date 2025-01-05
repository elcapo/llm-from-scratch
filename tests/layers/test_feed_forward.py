import torch
from scratch.layers.feed_forward import FeedForward
from scratch.gpt_config import GptConfig

def test_feed_forward():
    # Prepare
    torch.manual_seed(123)
    gpt_small = GptConfig(
        vocab_size = 64,
        context_length = 8,
        embedding_dimension = 4,
        head_count = 2,
        layer_count = 2,
        forward_layer_size = 4,
        drop_rate = 0.1,
        qkv_bias = True)
    feed_forward = FeedForward(gpt_small)
    x = torch.rand(2, 3, 4)
    # Act
    out = feed_forward(x)
    # Assert
    assert torch.allclose(
        out,
        torch.tensor([
            [
                [-0.0329, -0.2533,  0.0936, -0.1079],
                [-0.0054, -0.3291,  0.2460, -0.2613],
                [-0.0997, -0.2271,  0.1501, -0.1486]
            ],
            
            [
                [-0.0649, -0.2735,  0.2235, -0.1773],
                [ 0.0281, -0.3837,  0.3153, -0.2935],
                [ 0.0845, -0.3002,  0.1694, -0.0723]
            ]
        ]),
        atol=1e-4
    )
