from torch.nn import Embedding, Module
from .token_embeddings import TokenEmbeddings
from .positional_embeddings import PositionalEmbeddings

class InputEmbeddings(Module):
    def __init__(self, vocab_size: int, context_length: int, output_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.output_dim = output_dim

        self.token_embedding_layer = TokenEmbeddings(vocab_size, output_dim)
        self.positional_embedding_layer = PositionalEmbeddings(context_length, output_dim)
    
    def forward(self, x):
        token_embeddings = self.token_embedding_layer(x)
        positional_embeddings = self.positional_embedding_layer()
        return token_embeddings + positional_embeddings
