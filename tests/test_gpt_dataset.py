from scratch.gpt_dataset import GptDataset
from scratch.tokenizers.tiktoken_tokenizer import TiktokenTokenizer
from .fixtures import the_veredict

def test_gpt_dataset(the_veredict):
    dataset = GptDataset(the_veredict, TiktokenTokenizer(), 4)
    assert dataset[0][0] == [40, 367, 2885, 1464]
    assert dataset[0][1] == [367, 2885, 1464, 1807]
