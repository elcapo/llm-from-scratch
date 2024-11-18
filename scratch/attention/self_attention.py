from torch import rand, tensor
from torch.nn import Parameter
from .base_attention import BaseAttention
from ..normalizers.base_normalizer import BaseNormalizer
from ..normalizers.softmax_normalizer import SoftmaxNormalizer

class SelfAttention(BaseAttention):
    def __init__(self, d_in: int, d_out: int, normalizer: BaseNormalizer = SoftmaxNormalizer()):
        super().__init__()
        self.normalizer = normalizer
        self.W_query = Parameter(rand(d_in, d_out))
        self.W_key = Parameter(rand(d_in, d_out))
        self.W_value = Parameter(rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        scores = queries @ keys.T
        weights = self.normalizer.normalize(scores / keys.shape[-1]**0.5)
        return weights @ values
