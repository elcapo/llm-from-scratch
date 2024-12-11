from typing import Optional
import torch
from .batch_loss import BatchLoss
from ..gpt_model import GptModel

class LoaderLoss:
    def __init__(self, model: GptModel, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        self.device = device
        self.model = model
        self.batch_loss = BatchLoss(model, device)

    def calc(self, data_loader: torch.utils.data.DataLoader, num_batches: Optional[int] = None):
        total_loss = 0.

        if len(data_loader) == 0:
            return float("nan")

        if num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))

        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = self.batch_loss.calc(input_batch, target_batch)
                total_loss += loss.item()
            else:
                break

        return total_loss / num_batches
