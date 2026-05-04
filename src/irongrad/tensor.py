import irongrad_backend

class Tensor:
    
    def __init__(self, data, _children=()):
        self.data = [float(x) for x in data]
        self.grad = [0] * len(self.data)
        self._prev = set(_children)
        self._backward = lambda: None


    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = irongrad_backend.add_arrays(self.data, other.data)
        out = Tensor(out_data, _children=(self, other))

        def _backward():
            for i in range(len(self.data)):
                self.grad[i] += out.grad[i]
                other.grad[i] += out.grad[i]
        out._backward = _backward

        return out
    
    def backward(self):

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = [1.0] * len(self.data)

        for node in reversed(topo):
            node._backward()

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = irongrad_backend.mul_arrays(self.data, other.data)
        out = Tensor(out_data, _children=(self, other))

        def _backward():
            for i in range(len(self.data)):
                self.grad[i] += other.data[i] * out.grad[i]
                other.grad[i] += self.data[i] * out.grad[i]
        out._backward = _backward

        return out

    def sum(self):
        out = Tensor([irongrad_backend.sum_array(self.data)], _children=(self,))

        def _backward():
            for i in range(len(self.data)):
                self.grad[i] += out.grad[0]
        out._backward = _backward

        return out

    def relu(self):
        out = Tensor(irongrad_backend.relu_array(self.data), _children=(self,))

        def _backward():
            for i in range(len(self.data)):
                self.grad[i] += (1.0 if self.data[i] > 0 else 0.0) * out.grad[i]
        out._backward = _backward

        return out





    
    def __repr__(self) -> str:
        return f'Tensor(data={self.data}, grad={self.grad})'