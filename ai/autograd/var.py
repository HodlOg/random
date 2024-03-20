import numpy as np
from networkx import DiGraph, draw, circular_layout
from matplotlib import pyplot as plt
class Tensor:
    def __init__(self, value, _children=(), _op=''):
        self.value = np.array(value)
        self.grad = np.zeros_like(self.value, dtype=np.float64)  # Cast grad to float64
        self._backward = lambda: None
        self._children = _children
        self._op = _op
       
    def __add__(self, other):
        return add(self, other if isinstance(other, Tensor) else Tensor(other))
    def __mul__(self, other):
        return mul(self, other if isinstance(other, Tensor) else Tensor(other))
    def __log__(self):
        return log(self)
    def __pow__(self, other):
        return pow(self, other if isinstance(other, Tensor) else Tensor(other))
    def backward(self):
        return backward(self)
    def relu(self):
        return relu(self)
    def sigmoid(self):
        return sigmoid(self)
    def tanh(self):
        return tanh(self)
    def leaky_relu(self, alpha=0.01):
        return leaky_relu(self, alpha)
    def __matmul__(self, other):
        return matmul(self, other)
    def matmul(self, other):
        return matmul(self, other)
    
    # transpose
    def T(self):
        return Tensor(self.value.T)
    
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1
   
    def __repr__(self):
        return f"Value: {self.value}, Gradient: {self.grad}"
   
    def __str__(self):
        return f"Value: {self.value}, Gradient: {self.grad}"
    
    def printDag(self):
        graph = DiGraph()
        nodes = [self]
        while nodes:
            node = nodes.pop()
            graph.add_node(node)
            for child in node._children:
                graph.add_edge(node, child)
                nodes.append(child)
        draw(graph, with_labels=True, pos=circular_layout(graph))
        plt.show()

def relu(a):
    result = Tensor(np.maximum(0, a.value))
    result._backward = lambda: a.grad.__iadd__(result.grad * (a.value > 0))
    result._children = (a,)
    return result

def add(a, b):
    result = Tensor(a.value + b.value)
    result._backward = lambda: (a.grad.__iadd__(result.grad), b.grad.__iadd__(result.grad))
    result._children = (a, b)
    return result

def mul(a, b):
    result = Tensor(a.value * b.value)
    result._backward = lambda: (a.grad.__iadd__(b.value * result.grad), b.grad.__iadd__(a.value * result.grad))
    result._children = (a, b)
    return result

def matmul(a, b):
    result = Tensor(np.matmul(a.value, b.value))
    result._backward = lambda: (a.grad.__iadd__(np.matmul(result.grad, b.value.T)), b.grad.__iadd__(np.matmul(a.value.T, result.grad)))
    result._children = (a, b)
    return result

def log(a):
    result = Tensor(np.log(a.value))
    result._backward = lambda: a.grad.__iadd__(result.grad / a.value)
    result._children = (a,)
    return result

def pow(a, b):
    result = Tensor(a.value ** b.value)
    result._backward = lambda: a.grad.__iadd__(b.value * (a.value ** (b.value - 1)) * result.grad)
    result._children = (a, b)
    return result

def backward(self):
    topo = []
    visited = set()
   
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._children:
                build_topo(child)
            topo.append(v)
   
    build_topo(self)
    self.grad = np.ones_like(self.value, dtype=np.float64)  # cast initial grad to float64
   
    for v in reversed(topo):
        v._backward()

def sigmoid(a):
    result = Tensor(1 / (1 + np.exp(-a.value)))
    result._backward = lambda: a.grad.__iadd__(result.value * (1 - result.value) * result.grad)
    result._children = (a,)
    return result

def tanh(a):
    result = Tensor(np.tanh(a.value))
    result._backward = lambda: a.grad.__iadd__((1 - result.value**2) * result.grad)
    result._children = (a,)
    return result

def leaky_relu(a, alpha=0.01):
    result = Tensor(np.where(a.value > 0, a.value, alpha * a.value))
    result._backward = lambda: a.grad.__iadd__(result.grad * np.where(a.value > 0, 1, alpha))
    result._children = (a,)
    return result


def mse_loss(y_pred, y_true):
    result = Tensor(np.mean((y_pred.value - y_true.value)**2))
    result._backward = lambda: y_pred.grad.__iadd__(2 * (y_pred.value - y_true.value) / y_pred.value.size)
    result._children = (y_pred,)
    return result

class Layer:
    def __init__(self):
        self.params = []

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = Tensor(np.random.randn(input_size, output_size))
        self.bias = Tensor(np.zeros(output_size))
        self.params = [self.weights, self.bias]

    def forward(self, inputs):
        self.inputs = inputs
        return matmul(inputs, self.weights) + self.bias

    def backward(self, grad):
        print("grad:", grad)
        print("inputs:", self.inputs.T())
        self.weights.grad += matmul(self.inputs.T(), grad)
        print("weights.grad:", self.weights.grad)
        self.bias.grad += np.sum(grad, axis=0)
        return matmul(grad, self.weights.T)
    
class Model:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            print("grad:", grad)
            grad = layer.backward(grad)

    def params(self):
        return [param for layer in self.layers for param in layer.params]

model = Model()
model.add(Dense(2, 4))
model.add(Dense(4, 1))

x = Tensor([[1, 2], [3, 4]])
y_true = Tensor([[0], [1]])

y_pred = model.forward(x)
print(y_pred)
loss = mse_loss(y_pred, y_true)
