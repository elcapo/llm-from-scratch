# Build a Large Language Model from Scratch

## Coding attention mechanisms

### Normalizers

As part of this chapter we'll make use of different normalizations. To orchestrate them all, there is a `BaseNormalizer` class which implements a single `normalize` method:

```python
from torch import tensor

def normalize(self, inputs: tensor) -> tensor:
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
