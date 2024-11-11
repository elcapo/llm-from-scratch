from torch import tensor, equal
from scratch.gpt_dataset import GptDataset
from scratch.tokenizers.tiktoken_tokenizer import TiktokenTokenizer
from .fixtures import the_veredict

def test_gpt_dataset(the_veredict):
    dataset = GptDataset(the_veredict, TiktokenTokenizer(), 4)
    assert equal(dataset[0][0], tensor([40, 367, 2885, 1464]))
    assert equal(dataset[0][1], tensor([367, 2885, 1464, 1807]))
