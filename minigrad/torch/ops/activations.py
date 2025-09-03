# ops/activations.py
import numpy as np
from minigrad.tensor import Tensor
from minigrad.torch.ops.basic_ops import OpsFunction


class Sigmoid(OpsFunction):
    label = "sigmoid"
    @staticmethod
    def forward(ctx, a):
        out = 1 / (1 + np.exp(-a))
        ctx.save_for_tensors(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        return grad_output * out * (1 - out)


class Tanh(OpsFunction):
    label = "tanh"
    @staticmethod
    def forward(ctx, a):
        out = np.tanh(a)
        ctx.save_for_tensors(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        return grad_output * (1 - out ** 2)


class Relu(OpsFunction):
    
    label = "relu"
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_tensors(a)
        return np.maximum(0, a)

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        return grad_output * (a > 0)

# ---------- Functional API ---------- #

def sigmoid(t: Tensor) -> Tensor:
    return Sigmoid.apply(t)


def tanh(t: Tensor) -> Tensor:
    return Tanh.apply(t)


def relu(t: Tensor) -> Tensor:
    return Relu.apply(t)
