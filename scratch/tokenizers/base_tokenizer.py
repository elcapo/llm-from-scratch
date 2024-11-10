from typing import List
from .preprocessor import Preprocessor

class BaseTokenizer:
    """
    Provide a base class for implementing different tokenizers.
    """
    def __init__(self, source: List[str] = [], preprocessor: Preprocessor = Preprocessor()):
        vocabulary = sorted(set(source))
        vocabulary.extend(["<|endoftext|>", "<|unk|>"])
        self.str_to_int = {s: i for i, s in enumerate(vocabulary)}
        self.int_to_str = {i: s for i, s in enumerate(vocabulary)}
        self.preprocessor = preprocessor

    def encode(self, text: str) -> List[int]:
        pass
    
    def decode(self, tokens: List[int]) -> str:
        pass
