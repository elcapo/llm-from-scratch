class GptConfig:
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        emb_dim: int,
        n_heads: int,
        n_layers: int,
        drop_rate: float,
        qkv_bias: bool
    ):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.drop_rate = drop_rate
        self.qkv_bias = qkv_bias

GptConfig124M = GptConfig(
    vocab_size = 50257,
    context_length = 1024,
    emb_dim = 768,
    n_heads = 12,
    n_layers = 12,
    drop_rate = 0.1,
    qkv_bias = False,
)
