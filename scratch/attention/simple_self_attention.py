from typing import Optional
import torch
from .base_attention import BaseAttention
from ..normalizers.base_normalizer import BaseNormalizer
from ..normalizers.softmax_normalizer import SoftmaxNormalizer
class SimpleSelfAttention(BaseAttention):
    def __init__(self, d_in: int, normalizer: Optional[BaseNormalizer]=None):
        super().__init__()
        if normalizer is None:
            self.normalizer = SoftmaxNormalizer()

    def get_scores(self, inputs: torch.tensor) -> torch.tensor:
        return inputs @ inputs.T
    
    def get_weights(self, scores: torch.tensor) -> torch.tensor:
        return self.normalizer(scores)
    
    def get_context_vectors(self, inputs: torch.tensor, weights: torch.tensor) -> torch.tensor:
        return weights @ inputs

    def forward(self, x: torch.tensor) -> torch.tensor:
        scores = self.get_scores(x)
        weights = self.get_weights(scores)
        return self.get_context_vectors(x, weights)
