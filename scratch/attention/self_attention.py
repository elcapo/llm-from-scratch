from torch import tensor
from torch.nn import Linear
from .base_attention import BaseAttention
from ..normalizers.base_normalizer import BaseNormalizer
from ..normalizers.softmax_normalizer import SoftmaxNormalizer

class SelfAttention(BaseAttention):
    def __init__(self, d_in: int, d_out: int, qkv_bias=False, normalizer: BaseNormalizer=SoftmaxNormalizer()):
        super().__init__()
        self.normalizer = normalizer
        self.W_query = Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        scores = queries @ keys.T
        weights = self.normalizer.normalize(scores / keys.shape[-1]**0.5)
        return weights @ values