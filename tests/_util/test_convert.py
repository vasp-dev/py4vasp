# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._util.convert import text_to_string, to_complex
import numpy as np


def test_text_to_string():
    assert text_to_string(b"foo") == "foo"
    assert text_to_string("bar") == "bar"


def test_scalar_to_complex(Assert):
    scalar = np.array((0.0, 1.0))
    converted = to_complex(scalar)
    assert converted.shape == ()
    Assert.allclose(converted.real, scalar[0])
    Assert.allclose(converted.imag, scalar[1])


def test_vector_to_complex(Assert):
    vector = np.linspace(0, 9, 10).reshape(5, 2)
    converted = to_complex(vector)
    assert converted.shape == (5,)
    Assert.allclose(converted.real, vector[:, 0])
    Assert.allclose(converted.imag, vector[:, 1])


def test_matrix_to_complex(Assert):
    matrix = np.linspace(0, 29, 30).reshape(3, 5, 2)
    converted = to_complex(matrix)
    assert converted.shape == (3, 5)
    Assert.allclose(converted.real, matrix[:, :, 0])
    Assert.allclose(converted.imag, matrix[:, :, 1])
