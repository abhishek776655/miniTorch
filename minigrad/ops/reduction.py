# ops/reductions.py
from minigrad.ops.ops_base import OpsFunction
from minigrad.ops.basic_ops import to_numpy
import numpy as np

class Sum(OpsFunction):
    def __init__(self):
        super().__init__()
    label = "sum"
    @staticmethod
    def forward(ctx, a, axis=None, keepdims=False):
        ctx.save_for_tensors(a.data)
        ctx.axis = axis
        ctx.keepdims = keepdims
        return np.sum(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        a = to_numpy(a)
        grad_output = to_numpy(grad_output)
        # If keepdims is False, we need to expand grad_output
        if not ctx.keepdims and ctx.axis is not None:
            grad_output = np.expand_dims(grad_output, axis=ctx.axis)
        return OpsFunction.unbroadcast(grad_output * np.ones_like(a), a.shape)


class Mean(OpsFunction):
    label = "mean"
    @staticmethod
    def forward(ctx, a, axis=None, keepdims=False):
        ctx.save_for_tensors(a.data)
        ctx.axis = axis
        ctx.keepdims = keepdims
        if axis is None:
            ctx.size = a.size
        else:
            ctx.size = np.prod([a.shape[i] for i in np.atleast_1d(axis)])
        return np.mean(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        a = to_numpy(a)
        grad_output = to_numpy(grad_output)
        # If keepdims is False, we need to expand grad_output
        if not ctx.keepdims and ctx.axis is not None:
            grad_output = np.expand_dims(grad_output, axis=ctx.axis)
        return OpsFunction.unbroadcast(grad_output * np.ones_like(a) / ctx.size, a.shape)


class Max(OpsFunction):
    label = "max"
    @staticmethod
    def forward(ctx, a, axis=None, keepdims=False):
        out = np.max(a, axis=axis, keepdims=keepdims)
        ctx.save_for_tensors(a.data)
        ctx.axis = axis
        ctx.keepdims = keepdims
        ctx.out = out
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        a = to_numpy(a)
        grad_output = to_numpy(grad_output)
        # If keepdims is False and axis is specified, expand grad_output
        if not ctx.keepdims and ctx.axis is not None:
            grad_output = np.expand_dims(grad_output, axis=ctx.axis)
        # Create mask for maximum values
        mask = (a == np.max(a, axis=ctx.axis, keepdims=True))
        return OpsFunction.unbroadcast(mask * grad_output, a.shape)


class Min(OpsFunction):
    label = "min"
    @staticmethod
    def forward(ctx, a, axis=None, keepdims=False):
        out = np.min(a, axis=axis, keepdims=keepdims)
        ctx.save_for_tensors(a.data)
        ctx.axis = axis
        ctx.keepdims = keepdims
        ctx.out = out
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        a = to_numpy(a)
        grad_output = to_numpy(grad_output)
        # If keepdims is False and axis is specified, expand grad_output
        if not ctx.keepdims and ctx.axis is not None:
            grad_output = np.expand_dims(grad_output, axis=ctx.axis)
        # Create mask for minimum values
        mask = (a == np.min(a, axis=ctx.axis, keepdims=True))
        return OpsFunction.unbroadcast(mask * grad_output, a.shape)
