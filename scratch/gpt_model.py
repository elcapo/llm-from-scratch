import torch
from .gpt_config import GptConfig
from .layers.transformer_block import TransformerBlock
from .normalizers.layer_normalization import LayerNorm

class GptModel(torch.nn.Module):
    def __init__(self, config: GptConfig):
        super().__init__()
        self.debug = False
        self.config = config
        self.token_embeddings = torch.nn.Embedding(config.vocab_size, config.embedding_dimension)
        self.positional_embeddings = torch.nn.Embedding(config.context_length, config.embedding_dimension)
        self.dropout_embeddings = torch.nn.Dropout(config.drop_rate)
        self.transformer_blocks = torch.nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.layer_count)])
        self.final_normalization = LayerNorm(config.embedding_dimension)
        self.out_head = torch.nn.Linear(config.embedding_dimension, config.vocab_size, bias=False)

    def enable_debugging(self):
        self.debug = True

    def disable_debugging(self):
        self.debug = False

    def forward(self, inputs: torch.tensor) -> torch.tensor:
        self.debug_bag = { "inputs": inputs }

        batch_size, sequence_length = inputs.shape
        token_embeddings = self.token_embeddings(inputs)
        self.debug_bag["token_embeddings"] = token_embeddings

        positional_embeddings = self.positional_embeddings(
            torch.arange(sequence_length, device=inputs.device))
        self.debug_bag["positional_embeddings"] = positional_embeddings

        x = token_embeddings + positional_embeddings
        x = self.dropout_embeddings(x)
        self.debug_bag["dropout_embeddings"] = x

        x = self.transformer_blocks(x)
        self.debug_bag["transformer_blocks"] = x

        x = self.final_normalization(x)
        self.debug_bag["final_normalization"] = x

        logits = self.out_head(x)
        self.debug_bag["out_head"] = logits

        return logits
