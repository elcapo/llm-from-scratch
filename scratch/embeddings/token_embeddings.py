from torch.nn import Embedding, Module

class TokenEmbeddings(Module):
    def __init__(self, vocab_size: int, output_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.output_dim = output_dim

        self.embedding = Embedding(
            self.vocab_size,
            self.output_dim)
    
    def forward(self, x):
        return self.embedding(x)
