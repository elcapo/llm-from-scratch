from typing import List
import re

class Preprocessor:
    """
    Transforms a big chunk of text into a list of strings taking
    into account the separators that are usually present in human
    text readings.
    """
    def __init__(self, encoding_regex: str = r'([,.:;?_!"()\']|--|\s)', decoding_regex: str = r'\s+([,.:;?_!"()\'])'):
        self.encoding_regex = encoding_regex
        self.decoding_regex = decoding_regex

    def preprocess(self, text: str) -> List[str]:
        preprocessed = re.split(self.encoding_regex, text)
        return [item.strip() for item in preprocessed if item.strip()]
