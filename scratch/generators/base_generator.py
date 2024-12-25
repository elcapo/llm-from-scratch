import torch

class BaseGenerator:
    def generate(self, idx: torch.tensor) -> torch.tensor:
        pass

