import torch

class Gelu(torch.nn.Module):
    def forward(self, x: torch.tensor) -> torch.tensor:
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        ))
