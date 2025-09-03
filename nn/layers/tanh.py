from nn.module import Module
from nn import functional as F

class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.tanh(x)