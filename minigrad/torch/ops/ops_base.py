import numpy as np

class Context:
    def __init__(self):
        self.saved_tensors = ()
        
    def save_for_tensors(self, *tensors):
        self.saved_tensors = tensors
        
class OpsFunction:
    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise NotImplementedError
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError
    
    @classmethod
    def apply(cls, *tensors):
        from minigrad.tensor import Tensor 
        ctx = Context()
        raw_inputs = [t.data if isinstance(t,Tensor) else t for t in tensors]
        result = cls.forward(ctx, *raw_inputs)
        if not isinstance(result, np.ndarray):
            result = np.array(result)
    
        required_grad = any([t.requires_grad for t in tensors if isinstance(t,Tensor)])
        out = Tensor(result,requires_grad=required_grad)
        
        if required_grad:
            out._ctx = (cls, ctx, tensors)
        return out