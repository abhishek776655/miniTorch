from minigrad.ops.ops_base import OpsFunction
import numpy as np

def to_numpy(x):
    """Convert memoryview or any array-like to numpy array"""
    if hasattr(x, 'tobytes'):  # memoryview check
        return np.array(x)
    return x

class Add(OpsFunction):
    label = "add"
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_tensors(a.data, b.data)
        return a + b
    def backward(ctx, grad_output):
        a,b = map(to_numpy, ctx.saved_tensors)
        grad_output = to_numpy(grad_output)
        a_grad = OpsFunction.unbroadcast(1 * grad_output, a.shape)
        b_grad = OpsFunction.unbroadcast(1 * grad_output, b.shape)
        return a_grad, b_grad
        

class Sub(OpsFunction):
    label = "sub"
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_tensors(a.data, b.data)
        return a - b
    def backward(ctx, grad_output):
        a,b = map(to_numpy, ctx.saved_tensors)
        grad_output = to_numpy(grad_output)
        a_grad = OpsFunction.unbroadcast(1 * grad_output, a.shape)
        b_grad = OpsFunction.unbroadcast(-1 * grad_output, b.shape)
        return a_grad, b_grad

class Mul(OpsFunction):
    label = "mul"
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_tensors(a.data, b.data)
        return a * b
    def backward(ctx, grad_output):
        from minigrad.tensor import Tensor
        a, b = map(to_numpy, ctx.saved_tensors)
        grad_output = to_numpy(grad_output)
        if isinstance(grad_output, Tensor):
            raise NotImplementedError("grad_output should be a numpy array")
        a_grad = OpsFunction.unbroadcast(b * grad_output, a.shape)
        b_grad = OpsFunction.unbroadcast(a * grad_output, b.shape)
        return a_grad, b_grad

class Div(OpsFunction):
    label = "div"
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_tensors(a.data, b.data)
        return a / b
    
    def backward(ctx, grad_output):
        a, b = map(to_numpy, ctx.saved_tensors)
        grad_output = to_numpy(grad_output)
        a_grad = OpsFunction.unbroadcast((1 / b) * grad_output, a.shape)
        b_grad = OpsFunction.unbroadcast((-a / (b**2)) * grad_output, b.shape)
        return a_grad, b_grad

class MatMul(OpsFunction):
    label = "matmul"
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_tensors(a.data, b.data)
        return a @ b
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = map(to_numpy, ctx.saved_tensors)
        grad_output = to_numpy(grad_output)
            
        # For matrix multiplication, we need to handle batch dimensions
        if grad_output.ndim > 2:
            # Handle batched matmul
            a_grad = np.matmul(grad_output, b.swapaxes(-1, -2))
            b_grad = np.matmul(a.swapaxes(-1, -2), grad_output)
        else:
            # Standard matrix multiplication
            a_grad = grad_output @ np.array(b).T
            b_grad = np.array(a).T @ grad_output
        
        # Unbroadcast for batch dimensions
        a_grad = OpsFunction.unbroadcast(a_grad, a.shape)
        b_grad = OpsFunction.unbroadcast(b_grad, b.shape)
        return a_grad, b_grad

class Neg(OpsFunction):
    label = "neg"
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_tensors(a.data)
        return -a
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        grad_output = to_numpy(grad_output)
        a = to_numpy(a)
        return OpsFunction.unbroadcast(-grad_output, a.shape)
    
class Transpose(OpsFunction):
    label = "transpose"
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_tensors(a.data)
        return a.T
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        grad_output = to_numpy(grad_output)
        a = to_numpy(a)
        grad = grad_output.T
        # For transpose, the unbroadcast shape should be the transposed original shape
        return OpsFunction.unbroadcast(grad, a.T.shape)
    
class Exp(OpsFunction):
    label = "exp"
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_tensors(a.data)
        return np.exp(a)
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        grad_output = to_numpy(grad_output)
        a = to_numpy(a)
        grad = np.exp(a) * grad_output
        return OpsFunction.unbroadcast(grad, a.shape)