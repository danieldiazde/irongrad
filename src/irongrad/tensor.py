import irongrad_backend as backend

class Tensor:
    
    def __init__(self, data, shape = None, _children=()):


        self.data = [float(data)] if isinstance(data, (int,float)) else [float(x) for x in data]

        size = 1
        for dim in shape:
            size *= dim
        
        if size != len(self.data):
            raise ValueError("Shape and data mismatch")

        self.shape = shape if shape else (len(self.data),)
        self.grad = [0] * len(self.data)
        self._prev = set(_children)
        self._backward = lambda: None


    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        self._assert_same_shape(other)

        out_data = backend.add_arrays(self.data, other.data)
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

        self._assert_same_shape(other)


        out_data = backend.mul_arrays(self.data, other.data)
        out = Tensor(out_data, _children=(self, other))

        def _backward():
            for i in range(len(self.data)):
                self.grad[i] += other.data[i] * out.grad[i]
                other.grad[i] += self.data[i] * out.grad[i]
        out._backward = _backward

        return out

    def sum(self):
        out = Tensor([backend.sum_array(self.data)], _children=(self,))

        def _backward():
            for i in range(len(self.data)):
                self.grad[i] += out.grad[0]
        out._backward = _backward

        return out

    def relu(self):
        out = Tensor(backend.relu_array(self.data), _children=(self,))

        def _backward():
            for i in range(len(self.data)):
                self.grad[i] += (1.0 if self.data[i] > 0 else 0.0) * out.grad[i]
        out._backward = _backward

        return out

    @property
    def size(self):
        return len(self.data)
    
    def _assert_same_shape(self, other):
        if self.shape != other.shape:
            raise ValueError(
                f'Shape Mismatch: {self.shape} VS {other.shape}'
            )

    
    def __repr__(self) -> str:
        return f'Tensor(data={self.data}, grad={self.grad})'