import torch

class LayerNorm(torch.nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(emb_dim))
        self.shift = torch.nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.tensor) -> torch.tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        epsilon = 1e-5
        norm_x = (x - mean) / torch.sqrt(var + epsilon)
        return self.scale * norm_x + self.shift
