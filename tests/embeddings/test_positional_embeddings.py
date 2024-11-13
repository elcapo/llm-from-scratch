import torch
from torch import tensor, randint, allclose
from scratch.embeddings.positional_embeddings import PositionalEmbeddings

def test_positional_embeddings_dimensions():
    torch.manual_seed(123)
    vocab_size=50257
    output_dim=256
    max_length=4
    context_length=max_length
    batch_size=8
    embedding_layer = PositionalEmbeddings(max_length, output_dim)
    random_input = randint(low=0, high=vocab_size-1, size=(batch_size, context_length))
    embeddings = embedding_layer.embed()
    assert embeddings.shape == torch.Size([4, 256])

def test_positional_embeddings():
    torch.manual_seed(123)
    embedding_layer = PositionalEmbeddings(context_length=2, output_dim=4)
    embeddings = embedding_layer.embed()
    expected = tensor([
        [-0.1115,  0.1204, -0.3696, -0.2404],
        [-1.1969,  0.2093, -0.9724, -0.7550]
    ])
    assert allclose(embeddings, expected, atol=1e-4)
