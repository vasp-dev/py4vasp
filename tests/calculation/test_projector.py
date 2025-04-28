# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import pytest

from py4vasp import exception
from py4vasp._calculation.projector import SPIN_PROJECTION, Projector


@pytest.fixture
def Sr2TiO4(raw_data):
    return Projector.from_data(raw_data.projector("Sr2TiO4"))


@pytest.fixture
def Fe3O4(raw_data):
    return Projector.from_data(raw_data.projector("Fe3O4"))


@pytest.fixture
def Ba2PbO4(raw_data):
    return Projector.from_data(raw_data.projector("Ba2PbO4"))


@pytest.fixture
def missing_orbitals(raw_data):
    return Projector.from_data(raw_data.projector("without_orbitals"))


@pytest.fixture
def projections():
    num_spins = 1
    num_atoms = 7
    num_orbitals = 10
    num_quantity = 25
    shape = (num_spins, num_atoms, num_orbitals, num_quantity)
    return np.arange(np.prod(shape)).reshape(shape)


def test_read_missing_orbitals(missing_orbitals):
    assert missing_orbitals.read() == {}


def test_read_Sr2TiO4(Sr2TiO4):
    assert Sr2TiO4.read() == {
        "atom": {
            "Sr": slice(0, 2),
            "Ti": slice(2, 3),
            "O": slice(3, 7),
            "1": slice(0, 1),
            "2": slice(1, 2),
            "3": slice(2, 3),
            "4": slice(3, 4),
            "5": slice(4, 5),
            "6": slice(5, 6),
            "7": slice(6, 7),
        },
        "orbital": {
            "s": slice(0, 1),
            "p": slice(1, 4),
            "py": slice(1, 2),
            "pz": slice(2, 3),
            "px": slice(3, 4),
            "d": slice(4, 9),
            "dxy": slice(4, 5),
            "dyz": slice(5, 6),
            "dz2": slice(6, 7),
            "dxz": slice(7, 8),
            "dx2y2": slice(8, 9),
            "f": slice(9, 16),
            "fy3x2": slice(9, 10),
            "fxyz": slice(10, 11),
            "fyz2": slice(11, 12),
            "fz3": slice(12, 13),
            "fxz2": slice(13, 14),
            "fzx2": slice(14, 15),
            "fx3": slice(15, 16),
        },
        "spin": {
            "total": slice(0, 1),
        },
    }


def test_read_Fe3O4(Fe3O4):
    assert Fe3O4.read() == {
        "atom": {
            "Fe": slice(0, 3),
            "O": slice(3, 7),
            "1": slice(0, 1),
            "2": slice(1, 2),
            "3": slice(2, 3),
            "4": slice(3, 4),
            "5": slice(4, 5),
            "6": slice(5, 6),
            "7": slice(6, 7),
        },
        "orbital": {
            "s": slice(0, 1),
            "p": slice(1, 2),
            "d": slice(2, 3),
            "f": slice(3, 4),
        },
        "spin": {
            "total": slice(0, 2),
            "up": slice(0, 1),
            "down": slice(1, 2),
        },
    }


def test_read_Ba2PbO4(Ba2PbO4):
    assert Ba2PbO4.read() == {
        "atom": {
            "Ba": slice(0, 2),
            "Pb": slice(2, 3),
            "O": slice(3, 7),
            "1": slice(0, 1),
            "2": slice(1, 2),
            "3": slice(2, 3),
            "4": slice(3, 4),
            "5": slice(4, 5),
            "6": slice(5, 6),
            "7": slice(6, 7),
        },
        "orbital": {
            "s": slice(0, 1),
            "p": slice(1, 2),
            "d": slice(2, 3),
            "f": slice(3, 4),
        },
        "spin": {
            "total": slice(0, 1),
            "sigma_x": slice(1, 2),
            "x": slice(1, 2),
            "sigma_1": slice(1, 2),
            "sigma_y": slice(2, 3),
            "y": slice(2, 3),
            "sigma_2": slice(2, 3),
            "sigma_z": slice(3, 4),
            "z": slice(3, 4),
            "sigma_3": slice(3, 4),
        },
    }


def test_Sr2TiO4_project(Sr2TiO4, projections, Assert):
    Sr_ref = np.sum(projections[0, 0:2, 1:4], axis=(0, 1))
    Ti_ref = projections[0, 2, 4]
    actual = Sr2TiO4.project(selection="Sr(p) 3(dxy)", projections=projections)
    Assert.allclose(actual["Sr_p"], Sr_ref)
    Assert.allclose(actual["Ti_1_dxy"], Ti_ref)
    assert SPIN_PROJECTION not in actual


def test_spin_projections(Fe3O4, projections, Assert):
    spin_projections = np.array([projections[0] + 1, projections[0] - 1])
    Fe_ref = np.sum(spin_projections[:, 0:3], axis=(1, 2))
    d_ref = np.sum(spin_projections[:, :, 2], axis=(1))
    O_pd_ref = np.sum(spin_projections[:, 3:7, 1:3], axis=(1, 2))
    O_ref = np.sum(spin_projections[:, 3:7], axis=(0, 1, 2))
    p_ref = np.sum(spin_projections[:, :, 1], axis=(0, 1))
    down_ref = np.sum(spin_projections[1], axis=(0, 1))
    actual = Fe3O4.project("Fe O(p + d) d O(total) p + down", spin_projections)
    print(actual.keys())
    Assert.allclose(actual["Fe_up"], Fe_ref[0])
    Assert.allclose(actual["Fe_down"], Fe_ref[1])
    Assert.allclose(actual["d_up"], d_ref[0])
    Assert.allclose(actual["d_down"], d_ref[1])
    Assert.allclose(actual["O_p_up + O_d_up"], O_pd_ref[0])
    Assert.allclose(actual["O_p_down + O_d_down"], O_pd_ref[1])
    Assert.allclose(actual["O_total"], O_ref)
    Assert.allclose(actual["p + down"], p_ref + down_ref)
    assert SPIN_PROJECTION not in actual


def test_noncollinear_projections(Ba2PbO4, projections, Assert):
    projections = np.add.outer(np.linspace(-2, 2, 4), np.squeeze(projections))
    Pb_ref = np.sum(projections[0, 2], axis=0)
    total_ref = np.sum(projections[0], axis=(0, 1))
    p_x_ref = np.sum(projections[1, :, 1], axis=0)
    p_y_ref = np.sum(projections[2, :, 1], axis=0)
    BaPb_z_ref = np.sum(projections[3, 0:3], axis=(0, 1))
    xy_ref = np.sum(projections[1] - projections[2], axis=(0, 1))
    selection = "3 total p(x y) sigma_z(Ba + Pb) sigma_1 - sigma_2"
    actual = Ba2PbO4.project(selection, projections)
    Assert.allclose(actual["Pb_1"], Pb_ref)
    Assert.allclose(actual["total"], total_ref)
    Assert.allclose(actual["p_x"], p_x_ref)
    Assert.allclose(actual["p_y"], p_y_ref)
    Assert.allclose(actual["Ba_sigma_z + Pb_sigma_z"], BaPb_z_ref)
    Assert.allclose(actual["sigma_1 - sigma_2"], xy_ref, tolerance=100)
    expected = ["p_x", "p_y", "Ba_sigma_z + Pb_sigma_z", "sigma_1 - sigma_2"]
    assert actual[SPIN_PROJECTION] == expected


def test_noncollinear_projections(Ba2PbO4, projections, Assert):
    projections = np.add.outer(np.linspace(-2, 2, 4), np.squeeze(projections))
    Pb_ref = np.sum(projections[0, 2], axis=0)
    p_x_ref = np.sum(projections[1, :, 1], axis=0)
    p_y_ref = np.sum(projections[2, :, 1], axis=0)
    BaPb_z_ref = np.sum(projections[3, 0:3], axis=(0, 1))
    xy_ref = np.sum(projections[1] - projections[2], axis=(0, 1))
    actual = Ba2PbO4.project("3 p(x y) sigma_z(Ba + Pb) sigma_1 - sigma_2", projections)
    Assert.allclose(actual["Pb_1"], Pb_ref)
    Assert.allclose(actual["p_x"], p_x_ref)
    Assert.allclose(actual["p_y"], p_y_ref)
    Assert.allclose(actual["Ba_sigma_z + Pb_sigma_z"], BaPb_z_ref)
    Assert.allclose(actual["sigma_1 - sigma_2"], xy_ref, tolerance=100)


def test_missing_arguments_should_return_empty_dictionary(Sr2TiO4, projections):
    assert Sr2TiO4.project(selection="", projections=projections) == {}
    assert Sr2TiO4.project(selection=None, projections=projections) == {}


def test_missing_orbitals_project(missing_orbitals):
    with pytest.raises(exception.IncorrectUsage):
        missing_orbitals.project("any string", "any data")


def test_error_parsing(Sr2TiO4, projections):
    with pytest.raises(exception.IncorrectUsage):
        Sr2TiO4.project(selection="XX", projections=projections)
    with pytest.raises(exception.IncorrectUsage):
        number_instead_of_string = -1
        Sr2TiO4.project(selection=number_instead_of_string, projections=projections)
    with pytest.raises(exception.IncorrectUsage):
        Sr2TiO4.project(selection="up", projections=projections)


def test_incorrect_reading_of_projections(Sr2TiO4):
    with pytest.raises(exception.IncorrectUsage):
        Sr2TiO4.project("Sr", [1, 2, 3])
    with pytest.raises(exception.IncorrectUsage):
        Sr2TiO4.project("Sr", np.zeros(3))


def test_selections_Sr2TiO4(Sr2TiO4):
    p_orbitals = ["p", "px", "py", "pz"]
    d_orbitals = ["d", "dx2y2", "dxy", "dxz", "dyz", "dz2"]
    f_orbitals = ["f", "fx3", "fxyz", "fxz2", "fy3x2", "fyz2", "fz3", "fzx2"]
    assert Sr2TiO4.selections() == {
        "atom": ["Sr", "Ti", "O", "1", "2", "3", "4", "5", "6", "7"],
        "orbital": ["s", *p_orbitals, *d_orbitals, *f_orbitals],
        "spin": ["total"],
    }


def test_selections_Fe3O4(Fe3O4):
    assert Fe3O4.selections() == {
        "atom": ["Fe", "O", "1", "2", "3", "4", "5", "6", "7"],
        "orbital": ["s", "p", "d", "f"],
        "spin": ["total", "up", "down"],
    }


def test_selections_Ba2PbO4(Ba2PbO4):
    assert Ba2PbO4.selections() == {
        "atom": ["Ba", "Pb", "O", "1", "2", "3", "4", "5", "6", "7"],
        "orbital": ["s", "p", "d", "f"],
        "spin": [
            "total",
            "sigma_x",
            "sigma_y",
            "sigma_z",
            "x",
            "y",
            "z",
            "sigma_1",
            "sigma_2",
            "sigma_3",
        ],
    }


def test_selections_missing_orbitals(missing_orbitals):
    assert missing_orbitals.selections() == {}


def test_print(Sr2TiO4, format_):
    actual, _ = format_(Sr2TiO4)
    reference = """
projectors:
    atoms: Sr, Ti, O
    orbitals: s, py, pz, px, dxy, dyz, dz2, dxz, dx2y2, fy3x2, fxyz, fyz2, fz3, fxz2, fzx2, fx3
    """.strip()
    assert actual == {"text/plain": reference}


def test_missing_orbitals_print(missing_orbitals, format_):
    actual, _ = format_(missing_orbitals)
    assert actual == {"text/plain": "no projectors"}


def test_factory_methods(raw_data, check_factory_methods, projections):
    data = raw_data.projector("Sr2TiO4")
    parameters = {"project": {"selection": "Sr", "projections": projections}}
    check_factory_methods(Projector, data, parameters)
