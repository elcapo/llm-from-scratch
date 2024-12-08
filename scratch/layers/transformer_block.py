import torch
from ..gpt_config import GptConfig
from ..attention.multihead_attention import MultiHeadAttention
from ..normalizers.layer_normalization import LayerNorm
from .feed_forward import FeedForward

class TransformerBlock(torch.nn.Module):
    def __init__(self, config: GptConfig):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_in=config.embedding_dimension,
            d_out=config.embedding_dimension,
            context_length=config.context_length,
            num_heads=config.head_count,
            dropout=config.drop_rate,
            qkv_bias=config.qkv_bias)
        self.feed_forward = FeedForward(config)
        self.normalization1 = LayerNorm(config.embedding_dimension)
        self.normalization2 = LayerNorm(config.embedding_dimension)
        self.drop_shortcut = torch.nn.Dropout(config.drop_rate)

    def forward(self, x: torch.tensor) -> torch.tensor:
        shortcut = x
        x = self.normalization1(x)
        x = self.attention(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.normalization2(x)
        x = self.feed_forward(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
