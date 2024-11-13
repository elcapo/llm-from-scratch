import torch
from torch import tensor

class PositionalEmbeddings:
    def __init__(self, context_length: int, output_dim: int):
        self.context_length = context_length
        self.output_dim = output_dim

        self.embedding_layer = torch.nn.Embedding(
            self.context_length,
            self.output_dim)
    
    def embed(self):
        positions = torch.arange(self.context_length)
        return self.embedding_layer(positions)
