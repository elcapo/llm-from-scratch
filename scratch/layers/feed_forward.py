import torch
from ..activation.gelu import Gelu

class FeedForward(torch.nn.Module):
    def __init__(self, config: GptConfig):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(config.embedding_dimension, 4 * config.embedding_dimension),
            Gelu(),
            torch.nn.Linear(4 * config.embedding_dimension, config.embedding_dimension)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.layers(x)
