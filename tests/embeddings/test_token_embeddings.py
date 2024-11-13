import torch
from torch import tensor, randint, allclose
from scratch.embeddings.token_embeddings import TokenEmbeddings

def test_token_embeddings_dimensions():
    # Prepare
    torch.manual_seed(123)
    vocab_size=50257
    context_length=4
    output_dim=256
    batch_size=8
    # Act
    embedding_layer = TokenEmbeddings(vocab_size, output_dim)
    random_input = randint(low=0, high=vocab_size-1, size=(batch_size, context_length))
    embeddings = embedding_layer.embed(random_input)
    # Assert
    assert embeddings.shape == torch.Size([8, 4, 256])

def test_token_embeddings_one_dimension():
    # Prepare
    torch.manual_seed(123)
    # Act
    embedding_layer = TokenEmbeddings(vocab_size=6, output_dim=3)
    embeddings = embedding_layer.embed(tensor([3]))
    expected = tensor([[-0.4015,  0.9666, -1.1481]])
    # Assert
    assert allclose(embeddings, expected, atol=1e-4)

def test_token_embeddings_three_dimensions():
    # Prepare
    torch.manual_seed(123)
    # Act
    embedding_layer = TokenEmbeddings(vocab_size=6, output_dim=3)
    embeddings = embedding_layer.embed(tensor([2, 3, 5, 1]))
    # Assert
    expected = tensor([
        [ 1.2753, -0.2010, -0.1606],
        [-0.4015,  0.9666, -1.1481],
        [-2.8400, -0.7849, -1.4096],
        [ 0.9178,  1.5810,  1.3010],
    ])
    assert allclose(embeddings, expected, atol=1e-4)
  