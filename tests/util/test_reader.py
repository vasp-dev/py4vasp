# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

from py4vasp import exception
from py4vasp._util.reader import Reader


def test_reader():
    array = np.zeros(20)
    reader = Reader(array)
    with pytest.raises(exception.IncorrectUsage):
        reader[len(array) + 1]
