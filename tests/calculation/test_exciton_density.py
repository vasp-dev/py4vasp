# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import numpy as np
import pytest

from py4vasp import _config, exception, raw
from py4vasp._calculation.exciton_density import ExcitonDensity
from py4vasp._calculation.structure import Structure


@pytest.fixture
def exciton_density(raw_data):
    raw_density = raw_data.exciton_density()
    density = ExcitonDensity.from_data(raw_density)
    density.ref = types.SimpleNamespace()
    density.ref.structure = Structure.from_data(raw_density.structure)
    expected_charge = [component.T for component in raw_density.exciton_charge]
    density.ref.density = np.array(expected_charge)
    print(density.ref.density.shape)
    return density


@pytest.fixture
def empty_density(raw_data):
    raw_density = raw.ExcitonDensity(
        raw_data.structure("Sr2TiO4"), exciton_charge=raw.VaspData(None)
    )
    return ExcitonDensity.from_data(raw_density)


def test_read(exciton_density, Assert):
    actual = exciton_density.read()
    actual_structure = actual.pop("structure")
    Assert.same_structure(actual_structure, exciton_density.ref.structure.read())
    Assert.allclose(actual["charge"], exciton_density.ref.density)


def test_missing_data(empty_density):
    with pytest.raises(exception.NoData):
        empty_density.read()


@pytest.mark.parametrize("selection, indices", [(None, 0), ("2", 1), ("1, 3", (0, 2))])
def test_plot_selection(exciton_density, selection, indices, Assert):
    indices = np.atleast_1d(indices)
    if selection is None:
        view = exciton_density.plot()
    else:
        view = exciton_density.plot(selection)
    Assert.same_structure_view(view, exciton_density.ref.structure.plot())
    assert len(view.grid_scalars) == len(indices)
    for grid_scalar, index in zip(view.grid_scalars, indices):
        selected_exciton = exciton_density.ref.density[index]
        assert grid_scalar.label == str(index + 1)
        assert grid_scalar.quantity.ndim == 4
        Assert.allclose(grid_scalar.quantity, selected_exciton)
        assert len(grid_scalar.isosurfaces) == 1
        isosurface = grid_scalar.isosurfaces[0]
        assert isosurface.isolevel == 0.8
        assert isosurface.color == _config.VASP_COLORS["cyan"]
        assert isosurface.opacity == 0.6