import torch
from torch import tensor

torch.manual_seed(123)

class Embeddings:
    def __init__(self, vocab_size: int, output_dim: int):
        self.embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    
    def embed(self, input: tensor):
        return self.embedding_layer(input)
