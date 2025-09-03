from minigrad.ops.base import Function


class Add(Function):
    @staticmethod
    def forward(a, b):
        return a + b

class Sub(Function):
    @staticmethod
    def forward(a, b):
        return a - b

class Mul(Function):
    @staticmethod
    def forward(a, b):
        return a * b

class Div(Function):
    @staticmethod
    def forward(a, b):
        return a / b

class MatMul(Function):
    @staticmethod
    def forward(a, b):
        return a @ b
