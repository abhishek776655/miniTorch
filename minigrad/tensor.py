import numpy as np
from minigrad.ops.basic_ops import Add, Sub, Mul, Div, MatMul, Neg, Transpose, Exp
from minigrad.ops.unary import Pow
from minigrad.utils import topo_sort
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
    
     # --- Helpers ---
    @staticmethod
    def ensure_tensor(x):
        if isinstance(x, Tensor): return x
        return Tensor(x, requires_grad=False)
    
    # basic arithmetic operations
    def __add__(self, other):
        return Add.apply(self, Tensor.ensure_tensor(other))

    def __sub__(self, other):
        return Sub.apply(self, Tensor.ensure_tensor(other))

    def __mul__(self, other):
        return Mul.apply(self, Tensor.ensure_tensor(other))

    def __truediv__(self, other):
        return Div.apply(self, Tensor.ensure_tensor(other))

    def __matmul__(self, other):
        return MatMul.apply(self, Tensor.ensure_tensor(other))
        
    def __neg__(self):
        return Neg.apply(self)
    
    def T(self):
        return Transpose.apply(self)
    
    def exp(self):
        return Exp.apply(self)
    
    def __pow__(self, power, modulo=None):
        return Pow.apply(self, Tensor.ensure_tensor(power))
    
    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return Tensor(other).__sub__(self)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return Tensor(other).__truediv__(self)
    
        
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
        
        if self.grad is None:
            self.grad = Tensor(grad)
        else:
            self.grad = Tensor(self.grad.data + grad)
            
        for t in reversed(topo):
            if t._ctx:
                cls, ctx, inputs  = t._ctx
                if isinstance(t.grad, Tensor):
                    out_grad = t.grad.data
                grads = cls.backward(ctx, out_grad) # gradients produced by the operation
                if not isinstance(grads, tuple):
                    grads = (grads,)
                if len(grads) != len(inputs):
                    print(len(grads), len(inputs))
                    print(cls.label)
                    raise RuntimeError(f"returned wrong number of gradients")
                for input, grad in zip(inputs, grads):
                    if isinstance(input,Tensor) and input.requires_grad:
                        if input.grad is None:
                            input.grad = Tensor(grad) # assigining gradient to input of operations
                        else:
                            input.grad = Tensor(input.grad.data + grad)


    def detach(self):
        """
        Returns a new tensor with the same data but no gradient history.
        """
        return Tensor(self.data, requires_grad=False)

    def item(self):
        """
        Returns the value as a standard Python scalar.
        Only works if this tensor contains a single element.
        """
        if self.data.size != 1:
            raise ValueError("item() can only be called on a tensor with one element")
        return self.data.item()

    def numpy(self):
        """
        Returns the underlying NumPy array (detached).
        """
        return self.data
    
    def zero_grad(self):
        """
        Clears the gradient of this tensor.
        """
        self.grad = None