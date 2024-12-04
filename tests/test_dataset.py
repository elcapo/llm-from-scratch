import torch
from scratch.dataset import Dataset
from scratch.tokenizers.tiktoken_tokenizer import TiktokenTokenizer
from .fixtures import the_veredict

def test_dataset(the_veredict):
    # Prepare and act
    dataset = Dataset(the_veredict, TiktokenTokenizer(), 4)
    # Assert
    assert torch.equal(dataset[0][0], torch.tensor([40, 367, 2885, 1464]))
    assert torch.equal(dataset[0][1], torch.tensor([367, 2885, 1464, 1807]))
