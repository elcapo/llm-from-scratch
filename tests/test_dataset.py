from torch import tensor, equal
from scratch.dataset import Dataset
from scratch.tokenizers.tiktoken_tokenizer import TiktokenTokenizer
from .fixtures import the_veredict

def test_dataset(the_veredict):
    # Prepare and act
    dataset = Dataset(the_veredict, TiktokenTokenizer(), 4)
    # Assert
    assert equal(dataset[0][0], tensor([40, 367, 2885, 1464]))
    assert equal(dataset[0][1], tensor([367, 2885, 1464, 1807]))
