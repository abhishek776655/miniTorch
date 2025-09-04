from minigrad.tensor import Tensor
from minigrad.ops import basic_ops, unary, reductions

# ========= Elementwise =========
def add(x, y):
    return basic_ops.Add.apply(x, y)

def sub(x, y):
    return basic_ops.Sub.apply(x, y)

def mul(x, y):
    return basic_ops.Mul.apply(x, y)

def div(x, y):
    return basic_ops.Div.apply(x, y)

def pow(x, p):
    return unary.Pow.apply(x, p)

def exp(x):
    return unary.Exp.apply(x)

def log(x):
    return unary.Log.apply(x)

def sqrt(t: Tensor) -> Tensor:
    return unary.Sqrt.apply(t)

def square(t: Tensor) -> Tensor:
    return unary.Square.apply(t)

# ========= Reductions =========

def sum(t: Tensor,dim=None,keepdims=None) -> Tensor:
    return reductions.Sum.apply(t,axis=dim, keepdims=keepdims)

def mean(x, dim=None, keepdims=False):
    return reduce_ops.Mean.apply(x, axis=dim, keepdims=keepdims)

def max(x, dim=None, keepdims=False):
    return reduce_ops.Max.apply(x, axis=dim, keepdims=keepdims)

def min(x, dim=None, keepdims=False):
    return reduce_ops.Min.apply(x, axis=dim, keepdims=keepdims)

# ========= Linear algebra =========
def matmul(x, y):
    return basic_ops.MatMul.apply(x, y)


# ========= Shape ops =========
def reshape(x, *shape):
    return x.reshape(*shape)  # Tensor already implements reshape (wrap np.reshape)

def transpose(x, axes=None):
    return x.transpose(axes)  # Tensor already implements transpose