from nn.module import Module
from nn import functional as F

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.relu(x)