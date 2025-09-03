from minigrad.tensor import Tensor
from minigrad.build_graph import trace,draw_dot
from nn import functional as F
from nn.layers.linear import Linear
from nn.layers.relu import ReLU
from nn.layers.dropout import Dropout
from nn.layers.sequential import Sequential
from nn.optimizers.sgd import SGD
from nn.module import Module
from examples.MCP import MLP, train_mcp
import numpy as np

# Test basic operators
# x = Tensor([[1., 2.],
#             [3., 4.]], requires_grad=True)

# y = Tensor([10., 20.], requires_grad=True)
# c =x + y
# c.backward(Tensor([1., 1.],[1., 1.]))
# Display the computational graph
# dot =  draw_dot(c)

# Define a simple MLP

train_mcp()

# Visualize the computational graph
model = MLP()
x = Tensor(np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32))
y = Tensor(np.array([[0],[1],[1],[0]], dtype=np.float32))
out = model(x)
loss = F.sum(((out - y) ** 2))  # MSE loss
model.zero_grad()
loss.backward()
dot = draw_dot(loss)


