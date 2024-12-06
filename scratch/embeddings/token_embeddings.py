import torch

class TokenEmbeddings(torch.nn.Module):
    def __init__(self, vocab_size: int, output_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.output_dim = output_dim

        self.embedding = torch.nn.Embedding(
            self.vocab_size,
            self.output_dim)
    
    def forward(self, x):
        return self.embedding(x)
