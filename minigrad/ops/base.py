import numpy as np
class Function:
    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise NotImplementedError
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError
    
    @classmethod
    def apply(cls, *tensors):
        from minigrad.tensor import Tensor 
        raw_inputs = [t.data if isinstance(t,Tensor) else t for t in tensors]
        result = cls.forward(*raw_inputs)
        if not isinstance(result, np.ndarray):
            result = np.array(result)
        out = Tensor(result)
        return out