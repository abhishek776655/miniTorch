from nn.module import Module

class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
