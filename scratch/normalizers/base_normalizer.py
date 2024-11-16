from torch import tensor

class BaseNormalizer:
    def normalize(self, input: tensor) -> tensor:
        pass
