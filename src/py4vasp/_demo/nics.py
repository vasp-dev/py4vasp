# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw


def Sr2TiO4():
    structure = _demo.structure.Sr2TiO4()
    grid = (9, *_demo.GRID_DIMENSIONS)
    return raw.Nics(structure=structure, nics_grid=_demo.wrap_random_data(grid))


def Fe3O4():
    structure = _demo.structure.Fe3O4()
    seed_nics = 4782838
    seed_pos = 6375861
    positions_shape = (_demo.AXES, _demo.NUMBER_POINTS)
    nics_shape = (_demo.NUMBER_POINTS, _demo.AXES, _demo.AXES)
    nics_data = np.array(_demo.wrap_random_data(nics_shape, seed=seed_nics))
    # intentionally make values very small to check their output
    nics_data[4, 1, 0] = 1e-108
    nics_data[9, 0, 2] = -1e-15  # should be rounded
    nics_data[11, 2, 1] = 1e-14  # should still be there
    return raw.Nics(
        structure=structure,
        nics_points=raw.VaspData(nics_data),
        positions=_demo.wrap_random_data(positions_shape, seed=seed_pos),
    )
