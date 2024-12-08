import torch
from scratch.gpt_config import GptConfig
from scratch.gpt_model import GptModel

def test_gpt_model():
    # Prepare
    torch.manual_seed(123)
    config = GptConfig.small()
    model = GptModel(config)
    batch = torch.tensor([
        [6109, 3626, 6100, 345],
        [6109, 1110, 6622, 257]])
    # Act
    output = model(batch)
    # Assert
    assert output.shape == torch.Size((2, 4, 50257))
