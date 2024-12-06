from typing import Optional
import torch
from .base_attention import BaseAttention
from ..normalizers.base_normalizer import BaseNormalizer
from ..normalizers.adjusted_softmax_normalizer import AdjustedSoftmaxNormalizer

class CausalAttention(BaseAttention):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float=.5,
        qkv_bias=False,
        normalizer: Optional[BaseNormalizer]=None
    ):
        super().__init__()
        if normalizer is None:
            self.normalizer = AdjustedSoftmaxNormalizer(d_in)
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x: torch.tensor) -> torch.tensor:
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        scores = queries @ keys.transpose(1, 2)
        scores.masked_fill(self.mask.bool(), -torch.inf)
        weights = self.normalizer(scores)
        return weights @ values
