from typing import Optional
import torch
from ..gpt_model import GptModel

class BatchLoss:
    def __init__(self, model: GptModel, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        self.model = model
        self.device = device

    def calc(self, input_batch: torch.tensor, target_batch: torch.tensor):
        input_batch = input_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        logits = self.model(input_batch)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1),
            target_batch.flatten())
        return loss
