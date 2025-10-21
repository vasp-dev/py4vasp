# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import copy
import io
import itertools
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from py4vasp import exception
from py4vasp._third_party.view import View
from py4vasp._third_party.view.view import GridQuantity, IonArrow, Isosurface
from py4vasp._util import convert, import_

ase = import_.optional("ase")
ase_cube = import_.optional("ase.io.cube")


def base_input_view(is_structure):
    if is_structure:
        return {
            "elements": [["Sr", "Ti", "O", "O", "O"]],
            "lattice_vectors": [4 * np.eye(3)],
            "positions": [
                [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.5],
                    [0.0, 0.5, 0.5],
                    [0.5, 0.0, 0.5],
                    [0.5, 0.5, 0.0],
                ]
            ],
        }
    else:
        return {
            "elements": [["Ga", "As"], ["Ga", "As"]],
            "lattice_vectors": [
                2.8 * (np.ones((3, 3)) - np.eye(3)),
                2.9 * (np.ones((3, 3)) - np.eye(3)),
            ],
            "positions": [
                [
                    [0.0, 0.0, 0.0],
                    [0.25, 0.25, 0.25],
                ],
                [
                    [0.0, 0.0, 0.0],
                    [0.26, 0.24, 0.27],
                ],
            ],
        }
