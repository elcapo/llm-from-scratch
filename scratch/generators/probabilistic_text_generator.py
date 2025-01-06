from typing import Optional
import torch
from ..gpt_model import GptModel
from .base_generator import BaseGenerator

class ProbabilisticTextGenerator(BaseGenerator):
    def __init__(
        self,
        model: GptModel,
        max_new_tokens: int,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        eos_id: Optional[int] = None
    ):
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.eos_id = eos_id

    def generate(self, idx: torch.tensor) -> torch.tensor:
        context_length = self.model.config.context_length

        for _ in range(self.max_new_tokens):
            idx_cond = idx[:, -context_length:]
            with torch.no_grad():
                logits = self.model(idx_cond)
            logits = logits[:, -1, :]

            if self.model.debug:
                print()
                print("Input:", idx)
                print("Debug bag:", self.model.debug_bag)

            if self.top_k is not None:
                top_logits, _ = torch.topk(logits, self.top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(
                    logits < min_val,
                    torch.tensor(float('-inf')).to(logits.device),
                    logits)

            if self.temperature is not None and self.temperature > 0.0:
                logits = logits / self.temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                id_next = torch.argmax(logits, dim=-1, keepdim=True)

            if idx_next == self.eos_id:
                break

            idx = torch.cat((idx, idx_next), dim=1)
        return idx
