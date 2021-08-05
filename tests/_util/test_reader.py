from py4vasp._util.reader import Reader
import py4vasp.exceptions as exception
import numpy as np
import pytest


def test_reader():
    array = np.zeros(20)
    reader = Reader(array)
    with pytest.raises(exception.IncorrectUsage):
        reader[len(array) + 1]
