import torch
from scratch.attention.simple_self_attention import SimpleSelfAttention

def test_simple_self_attention_get_scores():
    # Prepare
    attention = SimpleSelfAttention(d_in=3)
    inputs = torch.tensor([
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55]])
    # Act
    scores = attention.get_scores(inputs)
    # Assert
    assert torch.allclose(
        scores[1],
        torch.tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865]),
        atol=1e-4)

def test_simple_self_attention_get_weights():
    # Prepare
    attention = SimpleSelfAttention(d_in=3)
    inputs = torch.tensor([
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55]])
    # Act
    scores = attention.get_scores(inputs)
    weights = attention.get_weights(scores)
    # Assert
    assert torch.allclose(
        weights[1],
        torch.tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581]),
        atol=1e-4)

def test_simple_self_attention_get_context_vectors():
    # Prepare
    attention = SimpleSelfAttention(d_in=3)
    inputs = torch.tensor([
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55]])
    # Act
    scores = attention.get_scores(inputs)
    weights = attention.get_weights(scores)
    context_vectors = attention.get_context_vectors(inputs, weights)
    # Assert
    assert torch.allclose(
        context_vectors[1],
        torch.tensor([0.4419, 0.6515, 0.5683]),
        atol=1e-4)

def test_simple_self_attention_compute():
    # Prepare
    attention = SimpleSelfAttention(d_in=3)
    inputs = torch.tensor([
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55]])
    # Act
    context_vectors = attention(inputs)
    # Assert
    assert torch.allclose(
        context_vectors[1],
        torch.tensor([0.4419, 0.6515, 0.5683]),
        atol=1e-4)