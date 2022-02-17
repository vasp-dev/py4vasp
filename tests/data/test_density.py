# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from unittest.mock import patch
from py4vasp.data import Density, Structure
import pytest
import numpy as np
import types
import py4vasp.data as data
import py4vasp.exceptions as exceptions


@pytest.fixture
def collinear_density(raw_data):
    raw_density = raw_data.density("Fe3O4 collinear")
    density = Density(raw_density)
    density.ref = types.SimpleNamespace()
    density.ref.structure = Structure(raw_density.structure).read()
    density.ref.charge = raw_density.charge[0]
    density.ref.magnetization = raw_density.charge[1]
    return density


@pytest.fixture
def charge_only_density(raw_data):
    raw_density = raw_data.density("Fe3O4 charge_only")
    density = Density(raw_density)
    density.ref = types.SimpleNamespace()
    density.ref.charge = raw_density.charge[0]
    return density


def test_read(collinear_density, Assert):
    actual = collinear_density.read()
    actual_structure = actual["structure"]
    reference_structure = collinear_density.ref.structure
    for key in actual_structure:
        if key in ("elements", "names"):
            assert actual_structure[key] == reference_structure[key]
        else:
            Assert.allclose(actual_structure[key], reference_structure[key])
    Assert.allclose(actual["charge"], collinear_density.ref.charge)
    Assert.allclose(actual["magnetization"], collinear_density.ref.magnetization)


def test_charge_plot(collinear_density, Assert):
    cm_init = patch.object(data.Viewer3d, "__init__", autospec=True, return_value=None)
    cm_cell = patch.object(data.Viewer3d, "show_cell")
    cm_arrows = patch.object(data.Viewer3d, "show_arrows_at_atoms")
    cm_surface = patch.object(data.Viewer3d, "show_isosurface")
    with cm_init as init, cm_cell as cell, cm_arrows as arrows, cm_surface as surface:
        collinear_density.plot()
        init.assert_called_once()
        cell.assert_called_once()
        surface.assert_called_once()
        args, kwargs = surface.call_args
    Assert.allclose(args[0], collinear_density.ref.charge)
    assert kwargs == {"isolevel": 0.2, "color": "yellow", "opacity": 0.6}


def test_magnetization_plot(collinear_density, Assert):
    cm_init = patch.object(data.Viewer3d, "__init__", autospec=True, return_value=None)
    cm_cell = patch.object(data.Viewer3d, "show_cell")
    cm_arrows = patch.object(data.Viewer3d, "show_arrows_at_atoms")
    cm_surface = patch.object(data.Viewer3d, "show_isosurface")
    with cm_init as init, cm_cell as cell, cm_arrows as arrows, cm_surface as surface:
        collinear_density.plot(selection="magnetization", isolevel=0.1, smooth=1)
        calls = surface.call_args_list
    assert len(calls) == 2
    _, kwargs = calls[0]
    assert kwargs == {"isolevel": 0.1, "color": "blue", "opacity": 0.6, "smooth": 1}
    _, kwargs = calls[1]
    assert kwargs == {"isolevel": 0.1, "color": "red", "opacity": 0.6, "smooth": 1}


def test_charge_only(charge_only_density, Assert):
    actual = charge_only_density.read()
    Assert.allclose(actual["charge"], charge_only_density.ref.charge)
    assert actual["magnetization"] is None


def test_missing_element(charge_only_density, Assert):
    with pytest.raises(exceptions.IncorrectUsage):
        charge_only_density.plot("unknown tag")
    with pytest.raises(exceptions.NoData):
        charge_only_density.plot("magnetization")


def test_color_specified_for_magnetism(collinear_density, Assert):
    with pytest.raises(exceptions.NotImplemented):
        collinear_density.plot("magnetization", color="brown")


def test_collinear_print(collinear_density, format_):
    actual, _ = format_(collinear_density)
    reference = """
density:
    structure: Fe3O4
    grid: 10, 12, 14
    spin polarized
    """.strip()
    assert actual == {"text/plain": reference}


def test_charge_only_print(charge_only_density, format_):
    actual, _ = format_(charge_only_density)
    reference = """
density:
    structure: Fe3O4
    grid: 10, 12, 14
    """.strip()
    assert actual == {"text/plain": reference}


def test_descriptor(collinear_density, check_descriptors):
    descriptors = {
        "_to_dict": ["to_dict", "read"],
        "_to_viewer3d": ["to_viewer3d", "plot"],
    }
    check_descriptors(collinear_density, descriptors)


def test_from_file(raw_data, mock_file, check_read):
    raw_density = raw_data.density("Fe3O4 collinear")
    with mock_file("density", raw_density) as mocks:
        check_read(Density, mocks, raw_density, default_filename="vaspwave.h5")
