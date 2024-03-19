import numpy as np

class Variable:
    def __init__(self, value, _children=(), _op=''):
        self.value = np.array(value)
        self.grad = np.zeros_like(self.value, dtype=np.float64)  # Cast grad to float64
        self._backward = lambda: None
        self._children = _children
        self._op = _op
       
    def __add__(self, other):
        return add(self, other if isinstance(other, Variable) else Variable(other))
    def __mul__(self, other):
        return mul(self, other if isinstance(other, Variable) else Variable(other))
    def __log__(self):
        return log(self)
    def __pow__(self, other):
        return pow(self, other if isinstance(other, Variable) else Variable(other))
    def backward(self):
        return backward(self)
    def relu(self):
        return relu(self)
       
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

def relu(a):
    result = Variable(np.maximum(0, a.value))
    result._backward = lambda: a.grad.__iadd__(result.grad * (a.value > 0))
    result._children = (a,)
    return result

def add(a, b):
    result = Variable(a.value + b.value)
    result._backward = lambda: (a.grad.__iadd__(result.grad), b.grad.__iadd__(result.grad))
    result._children = (a, b)
    return result

def mul(a, b):
    result = Variable(a.value * b.value)
    result._backward = lambda: (a.grad.__iadd__(b.value * result.grad), b.grad.__iadd__(a.value * result.grad))
    result._children = (a, b)
    return result

def log(a):
    result = Variable(np.log(a.value))
    result._backward = lambda: a.grad.__iadd__(result.grad / a.value)
    result._children = (a,)
    return result

def pow(a, b):
    result = Variable(a.value ** b.value)
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