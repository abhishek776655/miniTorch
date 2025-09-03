import numpy as np
from minigrad.tensor import Tensor
from nn.module import Module
from nn.parameter import Parameter
from nn.functional import linear

class Linear(Module):
    def __init__(self,in_features, out_features, bias = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(np.random.randn(in_features, out_features) / np.sqrt(in_features))
        if bias:
            self.bias = Parameter(np.zeros(out_features))
        else:
            self.bias = None
            
    def forward(self,x):
        return linear(x,self.weight,self.bias)