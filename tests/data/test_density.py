from unittest.mock import patch
from py4vasp.data import Density, _util
from py4vasp.raw import RawDensity
from .test_structure import raw_structure
from .test_topology import raw_topology
from . import current_vasp_version
import pytest
import numpy as np
import py4vasp.data as data
import py4vasp.exceptions as exceptions


@pytest.fixture
def raw_density(raw_structure):
    grid = (2, 10, 12, 14)
    density = RawDensity(
        version=current_vasp_version,
        structure=raw_structure,
        charge=np.arange(np.prod(grid)).reshape(grid),
    )
    return density


def test_from_file(raw_density, mock_file, check_read):
    with mock_file("density", raw_density) as mocks:
        check_read(Density, mocks, raw_density, default_filename="vaspwave.h5")


def test_read(raw_density, Assert):
    actual = Density(raw_density).read()
    lattice_vectors = actual["structure"]["lattice_vectors"]
    Assert.allclose(lattice_vectors, raw_density.structure.actual_cell)
    Assert.allclose(actual["structure"]["positions"], raw_density.structure.positions)
    assert actual["structure"]["elements"] == raw_density.structure.topology.elements
    Assert.allclose(actual["charge"], raw_density.charge[0])
    Assert.allclose(actual["magnetization"], raw_density.charge[1])


def test_plot(raw_density, Assert):
    cm_init = patch.object(data.Viewer3d, "__init__", autospec=True, return_value=None)
    cm_cell = patch.object(data.Viewer3d, "show_cell")
    cm_surface = patch.object(data.Viewer3d, "show_isosurface")
    with cm_init as init, cm_cell as cell, cm_surface as surface:
        density = Density(raw_density)
        density.plot()
        init.assert_called_once()
        cell.assert_called_once()
        surface.assert_called_once()
        args, kwargs = surface.call_args
    Assert.allclose(args[0], raw_density.charge[0])
    assert kwargs == {"isolevel": 0.2, "color": "yellow", "opacity": 0.6}
    #
    with cm_init as init, cm_cell as cell, cm_surface as surface:
        density = Density(raw_density)
        density.plot(quantity="magnetization", isolevel=0.1, smooth=1)
        calls = surface.call_args_list
    assert len(calls) == 2
    _, kwargs = calls[0]
    assert kwargs == {"isolevel": 0.1, "color": "blue", "opacity": 0.6, "smooth": 1}
    _, kwargs = calls[1]
    assert kwargs == {"isolevel": 0.1, "color": "red", "opacity": 0.6, "smooth": 1}


def test_charge_only(raw_density, Assert):
    raw_density.charge = raw_density.charge[:1]
    actual = Density(raw_density).read()
    Assert.allclose(actual["charge"], raw_density.charge[0])
    assert "magnetization" not in actual


def test_missing_element(raw_density, Assert):
    with pytest.raises(exceptions.IncorrectUsage):
        Density(raw_density).plot("unknown tag")
    raw_density.charge = raw_density.charge[:1]
    with pytest.raises(exceptions.NoData):
        Density(raw_density).plot("magnetization")


def test_color_specified_for_magnetism(raw_density, Assert):
    with pytest.raises(exceptions.NotImplemented):
        Density(raw_density).plot("magnetization", color="brown")


def test_print(raw_density):
    actual, _ = _util.format_(Density(raw_density))
    reference = """
density:
    structure: Sr2TiO4
    grid: 10, 12, 14
    spin polarized
    """.strip()
    assert actual == {"text/plain": reference}
    #
    raw_density.charge = np.reshape(raw_density.charge, (1, 16, 7, 30))
    actual, _ = _util.format_(Density(raw_density))
    reference = """
density:
    structure: Sr2TiO4
    grid: 16, 7, 30
    """.strip()
    assert actual == {"text/plain": reference}


def test_descriptor(raw_density, check_descriptors):
    density = Density(raw_density)
    descriptors = {
        "_to_dict": ["to_dict", "read"],
        "_to_viewer3d": ["to_viewer3d", "plot"],
    }
    check_descriptors(density, descriptors)
