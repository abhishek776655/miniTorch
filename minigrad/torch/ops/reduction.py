# ops/reductions.py
from minigrad.torch.ops.ops_base import OpsFunction
import numpy as np

class Sum(OpsFunction):
    label = "sum"
    @staticmethod
    def forward(ctx, a):
        ctx.shape = a.shape
        return np.array(a.sum())

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * np.ones(ctx.shape)


class Mean(OpsFunction):
    label = "mean"
    @staticmethod
    def forward(ctx, a):
        ctx.shape = a.shape
        ctx.size = a.size
        return np.array(a.mean())

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * np.ones(ctx.shape) / ctx.size


class Max(OpsFunction):
    label = "max"
    @staticmethod
    def forward(ctx, a):
        out = np.max(a)
        ctx.save_for_tensors(a, out)
        return np.array(out)

    @staticmethod
    def backward(ctx, grad_output):
        (a, out) = ctx.saved_tensors
        grad = (a == out).astype(a.dtype) * grad_output
        return grad


class Min(OpsFunction):
    label = "min"
    @staticmethod
    def forward(ctx, a):
        out = np.min(a)
        ctx.save_for_tensors(a, out)
        return np.array(out)

    @staticmethod
    def backward(ctx, grad_output):
        (a, out) = ctx.saved_tensors
        grad = (a == out).astype(a.dtype) * grad_output
        return grad
