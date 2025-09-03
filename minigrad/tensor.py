import numpy as np
from minigrad.ops.basic_ops import Add, Sub, Mul, Div, MatMul

class Tensor():
    def __init__(self, data, requires_grad=False):
        if not isinstance(data,np.ndarray):
            data = np.array(data)
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self.shape = data.shape
        self.dtype = data.dtype
    
    def __repr__(self):
        return f"Tensor(data = {self.data}, requires_grad = {self.requires_grad})"
    
    # basic arithmetic operations
    def __add__(self, other):
        return Add.apply(self, other)

    def __sub__(self, other):
        return Sub.apply(self, other)

    def __mul__(self, other):
        return Mul.apply(self, other)

    def __truediv__(self, other):
        return Div.apply(self, other)

    def __matmul__(self, other):
        return MatMul.apply(self, other)
    
