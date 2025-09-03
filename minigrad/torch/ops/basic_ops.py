from minigrad.torch.ops.ops_base import OpsFunction
import numpy as np

class Add(OpsFunction):
    label = "add"
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_tensors(a.data, b.data)
        return a + b
    def backward(ctx, grad_output):
        a_grad = 1 * grad_output
        b_grad = 1 * grad_output
        print("Add backward called")
        print("grad_output:", grad_output)
        print("a_grad:", a_grad)
        print("b_grad:", b_grad)
        return a_grad, b_grad
        

class Sub(OpsFunction):
    label = "sub"
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_tensors(a.data, b.data)
        return a - b
    def backward(ctx, grad_output):
        a_grad = 1 * grad_output
        b_grad = -1 * grad_output
        return a_grad, b_grad

class Mul(OpsFunction):
    label = "mul"
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_tensors(a.data, b.data)
        return a * b
    def backward(ctx, grad_output):
        from minigrad.tensor import Tensor
        a, b = ctx.saved_tensors
        if isinstance(grad_output, Tensor):
            raise NotImplementedError("grad_output should be a numpy array")
        a_grad = b * grad_output
        b_grad = a * grad_output
        return a_grad, b_grad

class Div(OpsFunction):
    label = "div"
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_tensors(a.data, b.data)
        return a / b
    
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        a_grad = (1 / b) * grad_output
        b_grad = (-a / (b**2)) * grad_output
        return a_grad, b_grad

class MatMul(OpsFunction):
    label = "matmul"
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_tensors(a.data, b.data)
        return a @ b
    
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        a_grad = grad_output @ b.T
        b_grad = a.T @ grad_output
        return a_grad, b_grad

class Neg(OpsFunction):
    label = "neg"
    @staticmethod
    def forward(a):
        return -a
    def backward(ctx, grad_output):
        return -grad_output
    
class Transpose(OpsFunction):
    label = "transpose"
    @staticmethod
    def forward(a):
        return a.T
    def backward(ctx, grad_output):
        return grad_output.T
    
class Exp(OpsFunction):
    label = "exp"
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_tensors(a.data)
        return np.exp(a)
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        return np.exp(a) * grad_output