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
