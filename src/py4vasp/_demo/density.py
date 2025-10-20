# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp import _demo, raw


def Sr2TiO4():
    structure = _demo.structure.Sr2TiO4()
    grid = (1, *_demo.GRID_DIMENSIONS)
    return raw.Density(structure=structure, charge=_demo.wrap_random_data(grid))


def Fe3O4(selection):
    structure = _demo.structure.Fe3O4()
    grid = (_demo.number_components(selection), *_demo.GRID_DIMENSIONS)
    return raw.Density(structure=structure, charge=_demo.wrap_random_data(grid))
