from torch import tensor
from torch.utils.data import Dataset, DataLoader
from .tokenizers.base_tokenizer import BaseTokenizer

class GptDataset(Dataset):
    def __init__(self, text: str, tokenizer: BaseTokenizer, max_length: int = 1024, stride: int = 1):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(tensor(input_chunk))
            self.target_ids.append(tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]
