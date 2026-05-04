from irongrad import Tensor

# ── helpers ──────────────────────────────────────────────────────────────────

def numerical_grad(f, tensor, idx, eps=1e-5):
    """Finite-difference gradient check for one element of a tensor.

    Computes (f(x+eps) - f(x-eps)) / 2eps. If our analytic grad matches this,
    the backward pass is correct.
    """
    orig = tensor.data[idx]

    tensor.data[idx] = orig + eps
    y_plus = f().data[0]

    tensor.data[idx] = orig - eps
    y_minus = f().data[0]

    tensor.data[idx] = orig
    return (y_plus - y_minus) / (2 * eps)

# ── add ───────────────────────────────────────────────────────────────────────

def test_add_backward():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    loss = (a + b).sum()
    loss.backward()
    # d(sum(a+b))/da_i = 1, same for b
    assert a.grad == [1.0, 1.0, 1.0]
    assert b.grad == [1.0, 1.0, 1.0]

# ── mul ───────────────────────────────────────────────────────────────────────

def test_mul_backward():
    a = Tensor([2.0, 3.0])
    b = Tensor([4.0, 5.0])
    loss = (a * b).sum()
    loss.backward()
    # d(sum(a*b))/da_i = b_i
    assert a.grad == [4.0, 5.0]
    assert b.grad == [2.0, 3.0]

# ── relu ──────────────────────────────────────────────────────────────────────

def test_relu_passes_positive_grad():
    a = Tensor([1.0, -2.0, 3.0])
    loss = a.relu().sum()
    loss.backward()
    # gradient passes through where input > 0, blocked where input <= 0
    assert a.grad == [1.0, 0.0, 1.0]

def test_relu_numerical():
    a = Tensor([0.5, -1.0, 2.0])
    for i in range(len(a.data)):
        analytic = None
        def f():
            return Tensor(a.data[:]).relu().sum()
        num = numerical_grad(lambda: Tensor(a.data[:]).relu().sum(), a, i)
        # recompute analytic
        a2 = Tensor(a.data[:])
        a2.relu().sum().backward()
        # analytic grad matches finite difference
        assert abs(a2.grad[i] - num) < 1e-4

# ── sum ───────────────────────────────────────────────────────────────────────

def test_sum_backward():
    a = Tensor([1.0, 2.0, 3.0])
    a.sum().backward()
    assert a.grad == [1.0, 1.0, 1.0]

# ── chained ops ───────────────────────────────────────────────────────────────

def test_chain_add_mul_relu():
    # loss = relu(a * b + c).sum()
    # verify with numerical gradients
    a = Tensor([1.0, -1.0, 2.0])
    b = Tensor([2.0,  3.0, 0.5])
    c = Tensor([0.5, -0.5, 1.0])

    def forward(ad, bd, cd):
        return (Tensor(ad) * Tensor(bd) + Tensor(cd)).relu().sum()

    loss = forward(a.data[:], b.data[:], c.data[:])
    a_copy = Tensor(a.data[:])
    b_copy = Tensor(b.data[:])
    c_copy = Tensor(c.data[:])
    out = (a_copy * b_copy + c_copy).relu().sum()
    out.backward()

    for i in range(len(a.data)):
        num_a = numerical_grad(lambda: forward(a.data[:], b.data[:], c.data[:]), a, i)
        num_b = numerical_grad(lambda: forward(a.data[:], b.data[:], c.data[:]), b, i)
        num_c = numerical_grad(lambda: forward(a.data[:], b.data[:], c.data[:]), c, i)
        assert abs(a_copy.grad[i] - num_a) < 1e-4, f"a grad mismatch at {i}"
        assert abs(b_copy.grad[i] - num_b) < 1e-4, f"b grad mismatch at {i}"
        assert abs(c_copy.grad[i] - num_c) < 1e-4, f"c grad mismatch at {i}"
