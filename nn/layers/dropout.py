import numpy as np
from nn.module import Module

class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.training = True

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        mask = (np.random.rand(*x.shape) > self.p).astype(x.dtype)
        return x * mask / (1.0 - self.p)