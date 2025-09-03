from minigrad.ops.activations import Sigmoid, Tanh, Relu
from minigrad.ops.unary import Exp, Log, Sqrt, Square
from minigrad.ops.reduction import Sum, Mean, Max, Min
from minigrad.tensor import Tensor

def sigmoid(t: Tensor) -> Tensor:
    return Sigmoid.apply(t)


def tanh(t: Tensor) -> Tensor:
    return Tanh.apply(t)


def relu(t: Tensor) -> Tensor:
    return Relu.apply(t)


def exp(t: Tensor) -> Tensor:
    return Exp.apply(t)


def log(t: Tensor) -> Tensor:
    return Log.apply(t)


def sqrt(t: Tensor) -> Tensor:
    return Sqrt.apply(t)


def square(t: Tensor) -> Tensor:
    return Square.apply(t)


def sum(t: Tensor) -> Tensor:
    return Sum.apply(t)


def mean(t: Tensor) -> Tensor:
    return Mean.apply(t)


def max(t: Tensor) -> Tensor:
    return Max.apply(t)


def min(t: Tensor) -> Tensor:
    return Min.apply(t)

# ----- NN Layers Functions API -------

def linear(input: Tensor, weight: Tensor, bias: Tensor = None) -> Tensor:
    out = input @ weight
    if bias is not None:
        out = out + bias
    return out