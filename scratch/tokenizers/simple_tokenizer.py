import re
from typing import List
from .base_tokenizer import BaseTokenizer
from .preprocessor import Preprocessor

class SimpleTokenizer(BaseTokenizer):
    """
    Implements a simple text tokenizer.
    """
    def encode(self, text: str) -> List[int]:
        preprocessed = self.preprocessor.preprocess(text)
        preprocessed = [s if s in self.str_to_int else "<|unk|>" for s in preprocessed]
        return [self.str_to_int[s] for s in preprocessed]

    def decode(self, tokens: List[int]) -> str:
        text = " ".join([self.int_to_str[i] for i in tokens])
        return re.sub(self.preprocessor.decoding_regex, r'\1', text)
