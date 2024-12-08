import torch
from scratch.gpt_config import GptConfig
from scratch.layers.transformer_block import TransformerBlock

def test_transformer_block():
    # Prepare
    gpt_small = GptConfig.small()
    transformer_block = TransformerBlock(gpt_small)
    x = torch.rand(2, 4, 768)
    # Act
    output = transformer_block(x)
    # Assert
    assert output.shape == torch.Size((2, 4, 768))
