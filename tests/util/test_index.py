# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

from py4vasp._util import index


@pytest.mark.parametrize("input, output", [("Sr", 1), ("Ti", 2), ("O", 3)])
def test_one_component(input, output):
    values = np.arange(10) ** 2
    map = {0: {"Sr": 1, "Ti": 2, "O": 3}}
    selector = index.Selector(map, values)
    assert selector[input] == values[output]


@pytest.mark.parametrize("input, output", [("A", [1, 2]), ("B", [3, 5, 7])])
def test_sum_over_selection(input, output):
    values = np.arange(10) ** 2
    map = {0: {"A": [1, 2], "B": slice(3, 8, 2)}}
    selector = index.Selector(map, values)
    assert selector[input] == np.sum(values[output])
