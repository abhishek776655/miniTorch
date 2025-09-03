class SGD:
    def __init__(self, parameters, lr = 0.01):
        self.params = list(parameters)
        self.lr = lr
        
    def step(self):
        for param in self.params:
            if param.grad is not None:
                param.data -= self.lr * param.grad.data
                
    def zero_grad(self):
        for param in self.params:
            param.grad = None