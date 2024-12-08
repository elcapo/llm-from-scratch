import torch
from scratch.tokenizers.tiktoken_tokenizer import TiktokenTokenizer
from scratch.gpt_config import GptConfig
from scratch.gpt_model import GptModel
from scratch.generators.simple_text_generator import SimpleTextGenerator

def test_simple_text_generator():
    # Prepare
    torch.manual_seed(123)
    tokenizer = TiktokenTokenizer()
    start_context = 'Hello, I am'
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    config = GptConfig.small()
    model = GptModel(config)
    model.eval()
    # Act
    generator = SimpleTextGenerator(
        model=model,
        max_new_tokens=6)
    output = generator.generate(encoded_tensor)
    # Assert
    output_tokens = output.squeeze(0).tolist()
    assert tokenizer.decode(output_tokens) == 'Hello, I amillasilaralon governed018ы'
