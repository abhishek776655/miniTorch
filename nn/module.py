from collections import OrderedDict
from minigrad.tensor import Tensor
class Module:
    def __init__(self):
        self._parameters = OrderedDict()
        self._modules = OrderedDict()
    
    def parameters(self):
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params
    
    def zero_grad(self):
        for params in self.parameters():
            params.grad = None
            
    def __setattr__(self, name, value):
        if isinstance(value, Tensor) and getattr(value, 'requires_grad', False):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError