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
