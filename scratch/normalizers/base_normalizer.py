from torch import tensor

class BaseNormalizer:
    def normalize(self, inputs: tensor) -> tensor:
        pass
