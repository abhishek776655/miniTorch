# 🧠 MiniTorch

_A minimal PyTorch-like deep learning framework built from scratch for learning purposes._

---

## 🚀 Features Implemented So Far

- **Core Tensor**

  - Wraps around `numpy.ndarray`
  - Supports autograd (automatic differentiation)
  - Builds computation graph dynamically
  - Basic operations: `+`, `-`, `*`, `/`, `@`, `exp`, `pow`, transpose, etc.
  - Supports broadcasting in forward and backward passes

- **Autograd Engine**

  - Tracks operations to build computation graph
  - Reverse-mode differentiation (backpropagation)
  - Topological sorting for gradient propagation
  - 🔍 `build_graph()` to visualize computation graph

## 🏗️ Neural Network Building Blocks

- `Tensor` with `.backward()`
- `Parameter` wrapper for learnable weights
- `Module` base class (like `nn.Module` in PyTorch)
- Layers implemented:

  - `Linear` (fully connected layer)
  - `Dropout` (training regularization)
  - Activation functions (`ReLU`, `Sigmoid`, etc.) in **functional API**

- **Optimizers**

  - `SGD` (Stochastic Gradient Descent) with learning rate

- **Losses**
  - MSE (`(y_pred - y_true) ** 2`)

## 🎯 Functional API

MiniTorch provides a functional API that offers a more flexible way to build neural networks. The functional API is stateless and doesn't maintain any internal state, making it great for custom operations and research.

### Getting Started

Import the functional API:

```python
from miniTorch.functional as miniTorch
from miniTorch.nn.functional as F
```

#### Element-wise Operations

````python
# Basic arithmetic
y = miniTorch.add(x1, x2)      # Element-wise addition
y = miniTorch.sub(x1, x2)      # Element-wise subtraction

# Mathematical functions
y = miniTorch.exp(x)           # Exponential
y = miniTorch.log(x)           # Natural logarithm
#### Reduction Operations
```python
# Reduction operations along specified dimensions
sum_x = miniTorch.sum(x, dim=1, keepdims=True)    # Sum along dimension 1

# Max along last dimension
min_x = miniTorch.min(x, dim=0, keepdims=True)    # Min with keepdims
````

## 🛠️ Installation

```bash
git clone https://github.com/abhishek776655/miniTorch.git
cd minigrad
pip install numpy
```

🔍 Visualizing the Computation Graph
from minigrad.torch.tensor import Tensor
from minigrad.torch.utils import build_graph

## Example: simple computation

```bash
a = Tensor([2.0], requires_grad=True)
b = Tensor([3.0], requires_grad=True)
c = a \* b + a
d = c.exp()
```

## Build and save graph

To view graph you need following installation

```bash
pip install graphviz matplotlib
```

code for saving and preview graph
here out is the name of output tensor

```bash
from miniTorch.minigrad.build_graph import draw_dot
dot = draw_dot(out)
dot.render("graph", format="png")  # saves graph.png
```

## 📖 API Reference

For a complete list of available functions and their detailed documentation, check the source code in `miniTorch/functional.py and miniTorch/nn/functional.py` or use Python's built-in `help()` function:

```python
import miniTorch.functional as F
import miniTorch.nn.functional as F
help(F.add)  # Get help for any function
```

## 🎯 Learning Goals

This project is for educational purposes only.
By building minigrad, you will learn:

How autograd engines work under the hood

How backpropagation computes gradients

How neural network layers, modules, and parameters are structured

How optimizers update weights

How frameworks like PyTorch/TensorFlow work internally

## 📌 Next Steps

Implement more layers (Conv2D, BatchNorm, etc.)

Add more optimizers (Adam, RMSProp, etc.)

Expand losses (CrossEntropyLoss)

Add dataset utilities (DataLoader)

Improve computation graph visualization

💡 This project is a playground to understand deep learning frameworks from scratch. Use it to build intuition before diving into PyTorch or TensorFlow!

## 🖼️ Computation Graph Preview

![Computation Graph Example](examples/graph.png)
