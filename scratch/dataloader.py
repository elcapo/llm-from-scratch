import torch
from .tokenizers.tiktoken_tokenizer import TiktokenTokenizer
from .tokenizers.base_tokenizer import BaseTokenizer
from .dataset import Dataset

def create_dataloader(
    text: str,
    tokenizer: BaseTokenizer = TiktokenTokenizer(),
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0
) -> torch.utils.data.DataLoader:
    dataset = Dataset(text, tokenizer, max_length, stride)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers)
