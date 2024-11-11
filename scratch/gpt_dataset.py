from torch import tensor
from torch.utils.data import Dataset, DataLoader
from .tokenizers.base_tokenizer import BaseTokenizer

class GptDataset(Dataset):
    def __init__(self, text: str, tokenizer: BaseTokenizer, max_length: int = 1024, stride: int = 1):
        self.ids = []

        token_ids = tokenizer.encode(text)
        for i in range(0, len(token_ids) - max_length + 1, stride):
            chunk = token_ids[i: i + max_length + 1]
            self.ids.append(tensor(chunk))
    
    def __len__(self):
        return len(self.ids) - 1
    
    def __getitem__(self, index):
        return self.ids[index], self.ids[index + 1]
