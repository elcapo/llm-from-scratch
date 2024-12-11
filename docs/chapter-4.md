# Build a Large Language Model from Scratch

## Implementing a GPT model from scratch to generate text

### Configuration

```python
from scratch.gpt_config import GptConfig

config = GptConfig.small()

config.context_length
```

This would return:

```python
1024 # config.context_length
```

### Normalization Layer

```python
import torch
from scratch.normalizers.layer_normalization import LayerNorm

torch.manual_seed(123)
batch_example = torch.rand(2, 3)
normalization = LayerNorm(emb_dim=3)
normalization(batch_example)
```

This would return:

```python
tensor([
    [-0.1327, -1.1475,  1.2802],
    [ 0.8105, -1.4087,  0.5982]
]) # normalization(batch_example)
```

### Gelu activation

```python
import torch
from scratch.activation.gelu import Gelu

gelu = Gelu()
x = torch.linspace(-3, 3, 5)
y = gelu(x)
```

This would return:

```python
tensor([-0.1588,  0.0000,  0.8412]) # y
```

### Feed forward

```python
import torch
from scratch.layers.feed_forward import FeedForward
from scratch.gpt_config import GptConfig

torch.manual_seed(123)
gpt_small = GptConfig(
    vocab_size = 64,
    context_length = 8,
    embedding_dimension = 2,
    head_count = 2,
    layer_count = 2,
    drop_rate = 0.1,
    qkv_bias = True)
feed_forward = FeedForward(gpt_small)
x = torch.rand(2, 2, 2)
```

```python
tensor([
    [
        [0.4545, 0.9737],
        [0.4606, 0.5159]
    ],
    [
        [0.4220, 0.5786],
        [0.9455, 0.8057]
    ]
]) # x
```

### Transformer block

```python
import torch
from scratch.gpt_config import GptConfig
from scratch.layers.transformer_block import TransformerBlock

gpt_small = GptConfig.small()
transformer_block = TransformerBlock(gpt_small)
x = torch.rand(2, 4, 768)
output = transformer_block(x)
```

```python
torch.Size((2, 4, 768)) # output.shape
```

### GPT Model

```python
import torch
from scratch.gpt_config import GptConfig
from scratch.gpt_model import GptModel

torch.manual_seed(123)
config = GptConfig.small()
model = GptModel(config)
batch = torch.tensor([
    [6109, 3626, 6100, 345],
    [6109, 1110, 6622, 257]])
output = model(batch)
```

```python
torch.Size((2, 4, 50257)) # output.shape
```

### References

#### Transformers

- [Visualizing transformers and attention](https://www.youtube.com/watch?v=KJtZARuO3JY) (2024) by Grant Sanderson.
- [Encoder vs decoder models](https://www.youtube.com/watch?v=XdGeVzDiYgg) (2024) by Emily McMilin.
- [The transformer architecture](https://www.youtube.com/watch?v=tstbZXNCfLY) (2021) by Sebastian Raschka.
- [The transformer architecture](https://www.youtube.com/watch?v=GhdB7UMtGqs) (2024) by Donato Capitella.

#### Shortcuts

When creating sequences that form our transformer blocks, we are adding shortcuts. To understand them, some literature about resudial networks might help. Here are some explainer videos:

- [Why residual connections work](https://www.youtube.com/watch?v=Gey9CG6R6w8) (2022) by DataMListic.
- [Residual networks and skip connections](https://www.youtube.com/watch?v=Q1JCrG1bJ-A) (2022) by Professor Bryce.

... and here are some relevant papers about the subject:

- [Deep residual learning for image recognition](https://arxiv.org/abs/1512.03385) (2015) by Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun.
- [Visualizing the loss landscape of neural nets](https://arxiv.org/abs/1712.09913) (2017) by Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer and Tom Goldstein.
