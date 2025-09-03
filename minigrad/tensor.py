import numpy as np
from minigrad.torch.ops.basic_ops import Add, Sub, Mul, Div, MatMul, Neg, Transpose, Exp
from minigrad.torch.utils import topo_sort
class Tensor():
    def __init__(self, data, requires_grad=False):
        if not isinstance(data,np.ndarray):
            data = np.array(data)
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self.shape = data.shape
        self.dtype = data.dtype
        self._ctx = None # context for autograd
    
    def __repr__(self):
        return f"Tensor(data = {self.data}, requires_grad = {self.requires_grad})"
    
    def __hash__(self):
        return id(self)
    
    def __eq__(self, other):
        return self is other
    
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
        
    def __neg__(self):
        return Neg.apply(self)
    
    def T(self):
        return Transpose.apply(self)
    
    def exp(self):
        return Exp.apply(self)
    
    def backward(self, grad=None):
        if self.requires_grad is False:
            return
        
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be specified for non-0-tensor")
            grad = np.ones_like(self.data, dtype=self.dtype)
        if isinstance(grad, Tensor):
            grad = grad.data
        
        topo = topo_sort(self)
        for t in reversed(topo):
            if t._ctx:
                cls, ctx, inputs  = t._ctx
                grads = cls.backward(ctx, grad)
                if not isinstance(grads, tuple):
                    grads = (grads,)
                if len(grads) != len(inputs):
                        raise RuntimeError(f"returned wrong number of gradients")
                for input, grad in zip(inputs, grads):
                    if isinstance(input,Tensor) and input.requires_grad:
                        if input.grad is None:
                            input.grad = Tensor(grad)
                        else:
                            input.grad = Tensor(input.grad.data + grad)

    