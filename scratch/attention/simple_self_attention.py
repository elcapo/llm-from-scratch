import torch
from torch import tensor
from .base_attention import BaseAttention

class SimpleSelfAttention(BaseAttention):
    def get_scores(self, inputs: torch.tensor) -> torch.tensor:
        return inputs @ inputs.T
    
    def get_weights(self, scores: torch.tensor) -> torch.tensor:
        return self.normalizer.normalize(scores)
    
    def get_context_vectors(self, inputs: tensor, weights: torch.tensor) -> torch.tensor:
        return weights @ inputs

    def forward(self, x: torch.tensor) -> torch.tensor:
        scores = self.get_scores(x)
        weights = self.get_weights(scores)
        return self.get_context_vectors(x, weights)
