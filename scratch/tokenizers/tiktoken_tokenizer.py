import tiktoken
from typing import List
from .base_tokenizer import BaseTokenizer
from .preprocessor import Preprocessor

class TiktokenTokenizer(BaseTokenizer):
    """
    Use OpenAI's tiktoken tokenizer.
    """
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding('gpt2')

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, allowed_special={'<|endoftext|>'})

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)
