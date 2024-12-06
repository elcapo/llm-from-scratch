import torch
from .base_attention import BaseAttention
from ..normalizers.base_normalizer import BaseNormalizer
from ..normalizers.softmax_normalizer import SoftmaxNormalizer

class MultiHeadAttention(BaseAttention):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float=.5,
        num_heads: int=4,
        qkv_bias=False,
        normalizer: BaseNormalizer=SoftmaxNormalizer()
    ):
        super().__init__()
        assert (d_out % num_heads == 0), 'The output size should be divisible by the number of heads'
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.normalizer = normalizer
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = torch.nn.Linear(d_out, d_out)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x: torch.tensor) -> torch.tensor:
        b, num_tokens, d_in = x.shape
        # Compute keys, queries and values
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        # Resize keys, queries and values
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        # Transpose keys, queries and values
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        # Compute the attention scores for each head
        scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        scores.masked_fill_(mask_bool, -torch.inf)
        # Compute the attention weights
        weights = self.normalizer(scores / keys.shape[-1]**0.5)
        weights = self.dropout(weights)
        # Compute the context vectors
        context_vectors = (weights @ values).transpose(1, 2)
        context_vectors = context_vectors.contiguous().view(b, num_tokens, self.d_out)
        context_vectors = self.out_proj(context_vectors)
        return context_vectors
