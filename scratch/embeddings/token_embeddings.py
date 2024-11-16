import torch
from torch import tensor

class TokenEmbeddings:
    def __init__(self, vocab_size: int, output_dim: int):
        self.vocab_size = vocab_size
        self.output_dim = output_dim

        self.embedding_layer = torch.nn.Embedding(
            self.vocab_size,
            self.output_dim)
    
    def embed(self, inputs: tensor):
        return self.embedding_layer(inputs)
