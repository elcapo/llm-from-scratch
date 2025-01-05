class GptConfig:
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embedding_dimension: int,
        head_count: int,
        layer_count: int,
        forward_layer_size: int,
        drop_rate: float,
        qkv_bias: bool
    ):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding_dimension = embedding_dimension
        self.head_count = head_count
        self.layer_count = layer_count
        self.forward_layer_size = forward_layer_size
        self.drop_rate = drop_rate
        self.qkv_bias = qkv_bias

    def small():
        return GptConfig(
            vocab_size = 50257,
            context_length = 1024,
            embedding_dimension = 768,
            head_count = 12,
            layer_count = 12,
            forward_layer_size = 4,
            drop_rate = 0.1,
            qkv_bias = True)

    def medium():
        return GptConfig(
            vocab_size = 50257,
            context_length = 1024,
            embedding_dimension = 1024,
            head_count = 16,
            layer_count = 24,
            forward_layer_size = 4,
            drop_rate = 0.1,
            qkv_bias = True)

    def large():
        return GptConfig(
            vocab_size = 50257,
            context_length = 1024,
            embedding_dimension = 1280,
            head_count = 20,
            layer_count = 36,
            forward_layer_size = 4,
            drop_rate = 0.1,
            qkv_bias = True)

    def xl():
        return GptConfig(
            vocab_size = 50257,
            context_length = 1024,
            embedding_dimension = 1600,
            head_count = 25,
            layer_count = 48,
            forward_layer_size = 4,
            drop_rate = 0.1,
            qkv_bias = True)
