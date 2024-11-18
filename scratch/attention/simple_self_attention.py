from torch import tensor
from .base_attention import BaseAttention

class SimpleSelfAttention(BaseAttention):
    def get_scores(self, inputs: tensor) -> tensor:
        return inputs @ inputs.T
    
    def get_weights(self, scores: tensor) -> tensor:
        return self.normalizer.normalize(scores)
    
    def get_context_vectors(self, inputs: tensor, weights: tensor) -> tensor:
        return weights @ inputs

    def forward(self, x):
        scores = self.get_scores(x)
        weights = self.get_weights(scores)
        return self.get_context_vectors(x, weights)
