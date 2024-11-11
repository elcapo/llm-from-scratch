from torch import tensor
from torch.utils.data import Dataset, DataLoader
from .tokenizers.base_tokenizer import BaseTokenizer

class GptDataset(Dataset):
    def __init__(self, text: str, tokenizer: BaseTokenizer, max_length: int = 1024, stride: int = 1):
        self.token_ids = tokenizer.encode(text)
        self.max_length = max_length
        self.stride = stride
    
    def __len__(self):
        return (len(self.token_ids) - self.max_length) // self.stride
    
    def __getitem__(self, index):
        if index + self.max_length + 1 >= len(self.token_ids):
            raise IndexError('list index out of range')
        step = index * self.stride
        return (
            self.token_ids[step : step + self.max_length],
            self.token_ids[step + 1 : step + self.max_length + 1]
        )