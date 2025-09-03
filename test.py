from minigrad.tensor import Tensor
from minigrad.torch.build_graph import trace,draw_dot
from minigrad.torch import functional as F

# Test basic operators
a = Tensor([1, 2, 3], requires_grad=True)
b = Tensor([4, 5, 6], requires_grad=True)
e = Tensor([7, 5, 6], requires_grad=True)

# Multiplication test with backprop
c = a * b

d = c + b
f = d * e
g = F.relu(f)
h = F.sigmoid(g)
h.backward(Tensor([1., 1., 1.]))
# Display the computational graph
dot =  draw_dot(h)
