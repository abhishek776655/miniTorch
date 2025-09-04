from minigrad.ops.activations import Sigmoid, Tanh, Relu
from minigrad.tensor import Tensor

def sigmoid(t: Tensor) -> Tensor:
    return Sigmoid.apply(t)


def tanh(t: Tensor) -> Tensor:
    return Tanh.apply(t)


def relu(t: Tensor) -> Tensor:
    return Relu.apply(t)


# ----- NN Layers Functions API -------

def linear(input: Tensor, weight: Tensor, bias: Tensor = None) -> Tensor:
    out = input @ weight
    if bias is not None:
        out = out + bias
    return out