class GptConfig:
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embedding_dimension: int,
        head_count: int,
        layer_count: int,
        drop_rate: float,
        qkv_bias: bool
    ):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding_dimension = embedding_dimension
        self.head_count = head_count
        self.layer_count = layer_count
        self.drop_rate = drop_rate
        self.qkv_bias = qkv_bias

    def default():
        return GptConfig(
            vocab_size = 50257,
            context_length = 1024,
            embedding_dimension = 768,
            head_count = 12,
            layer_count = 12,
            drop_rate = 0.1,
            qkv_bias = False)
