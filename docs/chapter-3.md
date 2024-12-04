# Build a Large Language Model from Scratch

## Coding attention mechanisms

### Normalizers

As part of this chapter we'll make use of different normalizations. To orchestrate them all, there is a `BaseNormalizer` class which implements a single `normalize` method:

```python
import torch

def normalize(self, inputs: torch.tensor) -> torch.tensor:
    pass
```

This class isn't able to do anything useful on its own. Instead, it's meant to be extended by other classes which will actually implement each of the normalization methods.

#### Add-One

The `AddOneNormalizer` is the most naive normalizer. It simply divides each component of a vector by the sum of its elements.

```python
from scratch.normalizers.add_one_normalizer import AddOneNormalizer

normalizer = AddOneNormalizer()
inputs = tensor([1., 8., 1.])
normalizer.normalize(inputs)
```

This would return:

```python
tensor([0.1000, 0.8000, 0.1000]) # normalizer.normalize(inputs)
```

#### Softmax

The `SoftmaxNormalizer` follows the more recommended approach of calling Pytorch's native `softmax` normalizer, although its usage is similar.

```python
from scratch.normalizers.softmax_normalizer import SoftmaxNormalizer

normalizer = SoftmaxNormalizer()
inputs = tensor([1., 8., 1.])
normalizer.normalize(inputs)
```

This would return:

```python
tensor([9.1022e-04, 9.9818e-01, 9.1022e-04]) # normalizer.normalize(inputs)
```

### Attention Mechanisms

#### Simple Self-Attention

The simple self-attention class is a simplified version of the attention mechanism. It lacks trainable weights but it shows how the attention scores and weights are computed with clarity.

```python
import torch
from scratch.attention.simple_self_attention import SimpleSelfAttention

attention = SimpleSelfAttention()
inputs = torch.rand(12, 3)
context_vectors = attention(inputs)
```

As this class is intended to show case the attention mechanism in its simplest form, it offers methods that facilitate the inspection of the intermediate matrices.

```python
scores = attention.get_scores(inputs)
weights = attention.get_weights(scores)
context_vectors = attention.get_context_vectors(inputs, weights)
```

##### Normalization

As the rest of the classes that we will implement in this chapter, it accepts a `normalizer` on construction.

```python
from scratch.normalizers.add_one_normalizer import AddOneNormalizer

attention = SimpleSelfAttention(normalizer=AddOneNormalizer())
```

When the class is instantiated without a normalizer, a soft max normalization is used.

#### Self Attention

The self attention class basically an extension of the previous simple self-attention, only that it has trainable weights.

```python
import torch
from scratch.attention.self_attention import SelfAttention

attention = SelfAttention(d_in=3, d_out=2)
inputs = torch.rand(12, 3)
context_vectors = attention(inputs)
```

#### Causal Attention

The causal attention class adds mechanisms to ensure that the transformer can't see the future during training. Also, it implements a dropout mechanism to prevent overfitting.

Finally, this class has been extended so that it supports more than one input at a time, so that it's easier to work with batches.

```python
import torch
from scratch.attention.causal_attention import CausalAttention

attention = CausalAttention(d_in=3, d_out=2, context_length=12)

batch = torch.stack((
    torch.rand(12, 3),
    torch.rand(12, 3),
    torch.rand(12, 3),
), dim=0)

context_vectors = attention(batch)
```

#### Multi-Head Attention

```python
import torch
from scratch.attention.multihead_attention import MultiHeadAttention

attention = MultiHeadAttention(
    d_in=3,
    d_out=2,
    context_length=12,
    dropout=0.0,
    num_heads=2)

batch = torch.stack((
    torch.rand(12, 3),
    torch.rand(12, 3),
    torch.rand(12, 3),
), dim=0)

context_vectors = attention(batch)
```
