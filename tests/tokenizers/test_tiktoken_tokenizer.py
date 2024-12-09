import torch
from ..fixtures import the_veredict
from scratch.tokenizers.preprocessor import Preprocessor
from scratch.tokenizers.tiktoken_tokenizer import TiktokenTokenizer

def test_tiktoken_tokenizer_encode():
    # Prepare
    tokenizer = TiktokenTokenizer()
    # Act
    text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
    tokens = tokenizer.encode(text)
    # Assert
    assert tokens == [15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250, 8812, 2114, 286, 617, 34680, 27271, 13]

def test_tiktoken_tokenizer_decode(the_veredict):
    # Prepare
    tokenizer = TiktokenTokenizer()
    # Act
    text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
    tokens = tokenizer.encode(text)
    decoded_text = tokenizer.decode(tokens)
    # Assert
    assert decoded_text == "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."

def test_tiktoken_tokenizer_text_to_tokens():
    # Prepare
    tokenizer = TiktokenTokenizer()
    text = "Hello, here I am"
    # Act
    tokens = tokenizer.text_to_tokens(text)
    # Assert
    assert torch.allclose(
        tokens,
        torch.tensor([[15496, 11, 994, 314, 716]])
    )

def test_tiktoken_tokenizer_tokens_to_text():
    # Prepare
    tokenizer = TiktokenTokenizer()
    tokens = torch.tensor([[15496, 11, 994, 314, 716]])
    # Act
    text = tokenizer.tokens_to_text(tokens)
    # Assert
    assert text == "Hello, here I am"
