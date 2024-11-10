from ..fixtures import the_veredict
from scratch.tokenizers.preprocessor import Preprocessor

def test_preprocessor():
    # Prepare
    text = "Hello, world. Is this-- a test?"
    preprocessor = Preprocessor()
    
    # Act
    preprocessed = preprocessor.preprocess(text)

    # Assert
    assert preprocessed == ['Hello', ',', 'world', '.', 'Is', 'this', '--', 'a', 'test', '?']

def test_preprocess_the_veredict(the_veredict):
    # Prepare
    preprocessor = Preprocessor()
    preprocessed = preprocessor.preprocess(the_veredict)
    
    # Act
    vocabulary = sorted(set(preprocessed))

    # Assert
    assert len(vocabulary) == 1130
