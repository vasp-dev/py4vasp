# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np

from py4vasp._third_party.view import View


def test_structure_to_view():
    view = View(
        number_ion_types=[[1, 1, 3]],
        ion_types=[["Sr", "Ti", "O"]],
        lattice_vectors=[4 * np.eye(3)],
        positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ],
    )
    view.to_ngl()
