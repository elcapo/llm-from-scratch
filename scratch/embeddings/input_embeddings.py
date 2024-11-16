import torch
from torch import tensor
from .token_embeddings import TokenEmbeddings
from .positional_embeddings import PositionalEmbeddings

class InputEmbeddings:
    def __init__(self, vocab_size: int, context_length: int, output_dim: int):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.output_dim = output_dim

        self.token_embedding_layer = TokenEmbeddings(vocab_size, output_dim)
        self.positional_embedding_layer = PositionalEmbeddings(context_length, output_dim)
    
    def embed(self, inputs: tensor):
        token_embeddings = self.token_embedding_layer.embed(inputs)
        positional_embeddings = self.positional_embedding_layer.embed()
        return token_embeddings + positional_embeddings
