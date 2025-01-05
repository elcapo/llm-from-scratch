import torch
from ..activation.gelu import Gelu
from ..gpt_config import GptConfig

class FeedForward(torch.nn.Module):
    def __init__(self, config: GptConfig):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(
                config.embedding_dimension,
                config.forward_layer_size * config.embedding_dimension
            ),
            Gelu(),
            torch.nn.Linear(
                config.forward_layer_size * config.embedding_dimension,
                config.embedding_dimension
            )
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.layers(x)
