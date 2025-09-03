from minigrad.tensor import Tensor
from minigrad.build_graph import trace,draw_dot
from nn import functional as F
from nn.layers.linear import Linear
from nn.layers.relu import ReLU
from nn.layers.dropout import Dropout
from nn.layers.sequential import Sequential
from nn.optimizers.sgd import SGD
from nn.module import Module
import numpy as np

# Define a simple MLP
class MLP(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(2, 4)
        self.fc2 = Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_mcp():
    
    X_data = np.array([
        [0,0],
        [0,1],
        [1,0],
        [1,1]
    ], dtype=np.float32)

    y_data = np.array([
        [0],
        [1],
        [1],
        [0]
    ], dtype=np.float32)

    X = Tensor(X_data)
    y = Tensor(y_data)

    # Model + optimizer
    model = MLP()
    optimizer = SGD(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(2000):
        # Forward pass
        out = model(X)
        loss = F.sum(((out - y) ** 2))  # MSE loss
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss = {loss.data:.4f}")

    # Final predictions
    print("\nPredictions after training:")
    print(model(X).data)
