import torch
from typing import List
from .preprocessor import Preprocessor

class BaseTokenizer:
    """
    Provide a base class for implementing different tokenizers.
    """
    def encode(self, text: str) -> List[int]:
        pass
    
    def decode(self, tokens: List[int]) -> str:
        pass

    def text_to_tokens(self, text: str) -> torch.tensor:
        encoded = self.encode(text)
        return torch.tensor(encoded).unsqueeze(0)

    def tokens_to_text(self, tokens: torch.tensor) -> str:
        flat = tokens.squeeze(0)
        return self.decode(flat.tolist())
