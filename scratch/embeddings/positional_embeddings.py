import torch

class PositionalEmbeddings(torch.nn.Module):
    def __init__(self, context_length: int, output_dim: int):
        super().__init__()
        self.context_length = context_length
        self.output_dim = output_dim

        self.embedding = torch.nn.Embedding(
            self.context_length,
            self.output_dim)
    
    def forward(self):
        positions = torch.arange(self.context_length)
        return self.embedding(positions)
