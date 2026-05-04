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


    
    def __repr__(self) -> str:
        return f'Tensor(data={self.data}, grad={self.grad})'