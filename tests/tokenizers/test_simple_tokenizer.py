from ..fixtures import the_veredict
from scratch.tokenizers.preprocessor import Preprocessor
from scratch.tokenizers.simple_tokenizer import SimpleTokenizer

def test_simple_tokenizer_encode(the_veredict):
    # Prepare
    preprocessor = Preprocessor()
    vocabulary = preprocessor.preprocess(the_veredict)
    tokenizer = SimpleTokenizer(vocabulary)
    # Act
    text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace."
    tokens = tokenizer.encode(text)
    # Assert
    assert tokens == [11, 5, 357, 1128, 630, 977, 12, 10, 57, 990, 958, 986, 724, 990, 11, 7]

def test_simple_tokenizer_decode(the_veredict):
    # Prepare
    preprocessor = Preprocessor()
    vocabulary = preprocessor.preprocess(the_veredict)
    tokenizer = SimpleTokenizer(vocabulary)
    # Act
    text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace."
    tokens = tokenizer.encode(text)
    decoded_text = tokenizer.decode(tokens)
    # Assert
    assert decoded_text == "<|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>."
