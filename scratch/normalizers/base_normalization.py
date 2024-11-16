from torch import tensor

class BaseNormalization:
    def normalize(self, input: tensor) -> tensor:
        pass
