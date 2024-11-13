import torch
from torch import tensor, randint, allclose
from scratch.embeddings.input_embeddings import InputEmbeddings

def test_input_embeddings_dimensions():
    # Prepare
    torch.manual_seed(123)
    vocab_size=50257
    context_length=4
    output_dim=256
    batch_size=8
    # Act
    embedding_layer = InputEmbeddings(vocab_size, context_length, output_dim)
    random_input = randint(low=0, high=vocab_size-1, size=(batch_size, context_length))
    embeddings = embedding_layer.embed(random_input)
    # Assert
    assert embeddings.shape == torch.Size([8, 4, 256])
