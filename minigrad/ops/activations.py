# ops/activations.py
import numpy as np
from minigrad.tensor import Tensor
from minigrad.ops.basic_ops import OpsFunction, to_numpy


class Sigmoid(OpsFunction):
    label = "sigmoid"
    @staticmethod
    def forward(ctx, a):
        out = 1 / (1 + np.exp(-a))
        ctx.save_for_tensors(out, a.data)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, a = map(to_numpy, ctx.saved_tensors)
        grad_output = to_numpy(grad_output)
        grad = grad_output * out * (1 - out)
        return OpsFunction.unbroadcast(grad, a.shape)


class Tanh(OpsFunction):
    label = "tanh"
    @staticmethod
    def forward(ctx, a):
        out = np.tanh(a)
        ctx.save_for_tensors(out, a.data)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, a = map(to_numpy, ctx.saved_tensors)
        grad_output = to_numpy(grad_output)
        grad = grad_output * (1 - out ** 2)
        return OpsFunction.unbroadcast(grad, a.shape)


class Relu(OpsFunction):
    label = "relu"
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_tensors(a.data)
        return np.maximum(0, a)

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        # Convert memoryview to numpy array if needed
        if hasattr(a, 'tobytes'):
            a = np.array(a)
        if hasattr(grad_output, 'tobytes'):
            grad_output = np.array(grad_output)
            
        grad = grad_output * np.where(a > 0, 1.0, 0.0)
        return OpsFunction.unbroadcast(grad, a.shape)
