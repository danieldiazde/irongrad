import pytest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build")))

import irongrad_backend

def test_add_arrays_basic():
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]
    expected = [5.0, 7.0, 9.0]
    assert irongrad_backend.add_arrays(a, b) == expected

def test_add_arrays_mismatch():
    a = [1.0, 2.0]
    b = [1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        irongrad_backend.add_arrays(a, b)
