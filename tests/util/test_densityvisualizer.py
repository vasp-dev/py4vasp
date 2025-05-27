from py4vasp._util import density
from py4vasp._util import index
from py4vasp._calculation.structure import Structure

import numpy as np

import pytest


def test_view(raw_data, Assert):
    structure = Structure.from_data(raw_data.structure("Fe3O4"))
    data3d = np.ones(shape=(3,4,5,10))
    selector = index.Selector({3: {"spin up": 0, "spin down": 1}}, data3d)

    visualizer = density.Visualizer(structure, selector)
    view = visualizer.to_view([("spin up",), ("spin down",)])

    Assert.same_structure_view(structure.to_view(), view)
    Assert.allclose(view.grid_scalars[0].quantity, data3d.T[0])
    Assert.allclose(view.grid_scalars[1].quantity, data3d.T[1])

@pytest.mark.parametrize("supercell", [(2,3,2), 3, (2,5,1)])
def test_view_supercell(raw_data, supercell, Assert):
    structure = Structure.from_data(raw_data.structure("Fe3O4"))
    data3d = np.ones(shape=(3,4,5))
    selector = index.Selector({}, data3d)

    visualizer = density.Visualizer(structure, selector)
    view = visualizer.to_view([()], supercell=supercell)

    Assert.same_structure_view(structure.to_view(supercell=supercell), view)
    Assert.allclose(view.grid_scalars[0].quantity, data3d.T)