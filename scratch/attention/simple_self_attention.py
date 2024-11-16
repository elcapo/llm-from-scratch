from torch import empty, dot, tensor
from .base_attention import BaseAttention

class SimpleSelfAttention(BaseAttention):
    def get_scores(self, inputs: tensor) -> tensor:
        return inputs @ inputs.T
    
    def get_weights(self, scores: tensor) -> tensor:
        return self.normalizer.normalize(scores)
    
    def get_context_vectors(self, inputs: tensor, weights: tensor) -> tensor:
        return weights @ inputs

    def compute(self, inputs: tensor) -> tensor:
        scores = self.get_scores(inputs)
        weights = self.get_weights(scores)
        return self.get_context_vectors(inputs, weights)

