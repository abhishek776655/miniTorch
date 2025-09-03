from minigrad.tensor import Tensor

# Test basic operators
a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])

print("a:", a)
print("b:", b)

print("a + b =", a + b)
print("a - b =", a - b)
print("a * b =", a * b)
print("a / b =", a / b)

# Test matrix multiplication
x = Tensor([[1, 2], [3, 4]])
y = Tensor([[5, 6], [7, 8]])
print("x @ y =", x @ y)

