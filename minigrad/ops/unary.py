# ops/unary.py
import numpy as np
from minigrad.ops.ops_base import OpsFunction
from minigrad.ops.basic_ops import to_numpy


class Exp(OpsFunction):
    label = "exp"
    @staticmethod
    def forward(ctx, a):
        out = np.exp(a)
        ctx.save_for_tensors(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        out = to_numpy(out)
        grad_output = to_numpy(grad_output)
        return grad_output * out


class Log(OpsFunction):
    label = "log"
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_tensors(a)
        return np.log(a)

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        a = to_numpy(a)
        grad_output = to_numpy(grad_output)
        grad = grad_output / a
        return OpsFunction.unbroadcast(grad, a.shape)


class Sqrt(OpsFunction):
    label = "squre root"
    @staticmethod
    def forward(ctx, a):
        out = np.sqrt(a)
        ctx.save_for_tensors(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        out = to_numpy(out)
        grad_output = to_numpy(grad_output)
        return grad_output * (0.5 / out)


class Square(OpsFunction):
    label = "square"
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_tensors(a)
        return a ** 2

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        a = to_numpy(a)
        grad_output = to_numpy(grad_output)
        grad = grad_output * (2 * a)
        return OpsFunction.unbroadcast(grad, a.shape)

class Pow(OpsFunction):
    label = "power"
    @staticmethod
    def forward(ctx, a, power):
        ctx.save_for_tensors(a, power)
        return a ** power

    @staticmethod
    def backward(ctx, grad_output):
        a, power = map(to_numpy, ctx.saved_tensors)
        grad_output = to_numpy(grad_output)
        
        # Gradient with respect to base a
        grad_a = power * (a ** (power - 1)) * grad_output
        # Gradient with respect to exponent (power)
        grad_power = (a ** power) * np.log(a) * grad_output
        
        # Handle broadcasting for both gradients
        grad_a = OpsFunction.unbroadcast(grad_a, a.shape)
        grad_power = OpsFunction.unbroadcast(grad_power, power.shape)
        
        return grad_a, grad_power

