# ops/unary.py
import numpy as np
from minigrad.tensor import Tensor
from minigrad.torch.ops.ops_base import OpsFunction


class Exp(OpsFunction):
    label = "exp"
    @staticmethod
    def forward(ctx, a):
        out = np.exp(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        return grad_output * out


class Log(OpsFunction):
    label = "log"
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return np.log(a)

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        return grad_output / a


class Sqrt(OpsFunction):
    label = "squre root"
    @staticmethod
    def forward(ctx, a):
        out = np.sqrt(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        return grad_output * (0.5 / out)


class Square(OpsFunction):
    label = "square"
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return a ** 2

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        return grad_output * (2 * a)

