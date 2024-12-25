import torch
from ..gpt_model import GptModel
from .base_generator import BaseGenerator

class SimpleTextGenerator(BaseGenerator):
    def __init__(self, model: GptModel, max_new_tokens: int):
        self.model = model
        self.max_new_tokens = max_new_tokens

    def generate(self, idx: torch.tensor) -> torch.tensor:
        context_length = self.model.config.context_length

        for _ in range(self.max_new_tokens):
            idx_cond = idx[:, -context_length:]
            with torch.no_grad():
                logits = self.model(idx_cond)
            logits = logits[:, -1, :]
            probas = torch.softmax(logits, dim=-1)
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
