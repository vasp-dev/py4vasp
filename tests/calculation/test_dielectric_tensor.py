# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import pytest

from py4vasp import exception
from py4vasp._calculation.cell import Cell
from py4vasp._calculation.dielectric_tensor import (
    DielectricTensor,
    _calculate_2d_polarizability,
)
from py4vasp._util import check


@pytest.fixture
def dft_tensor(raw_data):
    expected_description = "including local field effects in DFT"
    return make_reference(raw_data, "dft with_ion", expected_description)


@pytest.fixture
def rpa_tensor(raw_data):
    expected_description = "including local field effects in RPA (Hartree)"
    return make_reference(raw_data, "rpa without_ion", expected_description)


@pytest.fixture
def scf_tensor(raw_data):
    expected_description = "including local field effects"
    return make_reference(raw_data, "scf with_ion", expected_description)


@pytest.fixture
def nscf_tensor(raw_data):
    expected_description = "excluding local field effects"
    return make_reference(raw_data, "nscf without_ion", expected_description)


@pytest.fixture
def tensor_with_slab_cell(raw_data):
    expected_description = "including local field effects in DFT"
    return make_reference(raw_data, "dft_slab with_ion", expected_description)


def make_reference(raw_data, method, expected_description):
    raw_tensor = raw_data.dielectric_tensor(method)
    tensor = DielectricTensor.from_data(raw_tensor)
    tensor.ref = types.SimpleNamespace()
    tensor.ref.clamped_ion = raw_tensor.electron
    if raw_tensor.ion.is_none():
        tensor.ref.relaxed_ion = None
    else:
        tensor.ref.relaxed_ion = raw_tensor.ion + raw_tensor.electron
    tensor.ref.independent_particle = raw_tensor.independent_particle
    tensor.ref.polarizability_2d = None
    tensor.ref.polarizability_2d_ionic = None
    if not (check.is_none(raw_tensor.cell)):
        tensor.ref.cell = Cell.from_data(raw_tensor.cell)

        tensor.ref.polarizability_2d = _calculate_2d_polarizability(
            tensor.ref.relaxed_ion, tensor.ref.cell
        )
        if not check.is_none(raw_tensor.ion):
            tensor.ref.polarizability_2d_ionic = _calculate_2d_polarizability(
                raw_tensor.ion, tensor.ref.cell
            )
        tensor.ref.polarizability_2d_electronic = _calculate_2d_polarizability(
            raw_tensor.electron, tensor.ref.cell
        )
    else:
        tensor.ref.polarizability_2d_electronic = None
    tensor.ref.method = method.split()[0]
    tensor.ref.expected_description = expected_description
    return tensor


def test_read_dft_tensor(dft_tensor, Assert):
    check_read_dielectric_tensor(dft_tensor, Assert)


def test_read_rpa_tensor(rpa_tensor, Assert):
    check_read_dielectric_tensor(rpa_tensor, Assert)


def test_read_scf_tensor(scf_tensor, Assert):
    check_read_dielectric_tensor(scf_tensor, Assert)


def test_read_nscf_tensor(nscf_tensor, Assert):
    check_read_dielectric_tensor(nscf_tensor, Assert)


def check_read_dielectric_tensor(dielectric_tensor, Assert):
    for method in (dielectric_tensor.read, dielectric_tensor.to_dict):
        actual = method()
        reference = dielectric_tensor.ref
        Assert.allclose(actual["clamped_ion"], reference.clamped_ion)
        Assert.allclose(actual["relaxed_ion"], reference.relaxed_ion)
        Assert.allclose(actual["independent_particle"], reference.independent_particle)
        assert actual["method"] == reference.method


def test_unknown_method(raw_data):
    raw_tensor = raw_data.dielectric_tensor("unknown_method with_ion")
    with pytest.raises(exception.NotImplemented):
        DielectricTensor.from_data(raw_tensor).print()


def test_print_dft_tensor(dft_tensor, format_):
    actual, _ = format_(dft_tensor)
    check_print_dielectric_tensor(actual, dft_tensor.ref)


def test_print_rpa_tensor(rpa_tensor, format_):
    actual, _ = format_(rpa_tensor)
    check_print_dielectric_tensor(actual, rpa_tensor.ref)


def test_print_scf_tensor(scf_tensor, format_):
    actual, _ = format_(scf_tensor)
    check_print_dielectric_tensor(actual, scf_tensor.ref)


def test_print_dft_tensor(nscf_tensor, format_):
    actual, _ = format_(nscf_tensor)
    check_print_dielectric_tensor(actual, nscf_tensor.ref)


def check_print_dielectric_tensor(actual, reference):
    if reference.relaxed_ion is None:
        relaxed_ion = ""
    else:
        relaxed_ion = """\
                      relaxed-ion
          9.000000    11.000000    13.000000
         15.000000    17.000000    19.000000
         21.000000    23.000000    25.000000
"""
    expected = f"""
Macroscopic static dielectric tensor (dimensionless)
  {reference.expected_description}
------------------------------------------------------
                      clamped-ion
          0.000000     1.000000     2.000000
          3.000000     4.000000     5.000000
          6.000000     7.000000     8.000000
{relaxed_ion}
""".strip()
    assert actual == {"text/plain": expected}


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.dielectric_tensor("dft with_ion")
    check_factory_methods(DielectricTensor, data)


def _check_to_database(tensor, Assert):
    actual = tensor._read_to_database()
    db_dict = actual["dielectric_tensor:default"]

    assert db_dict["method"] == tensor.ref.method
    import numpy as np

    # check tensors
    relaxed_ion_expected_list = [9.0, 17.0, 25.0, 13.0, 21.0, 17.0]
    ionic_expected_list = [9.0, 13.0, 17.0, 11.0, 15.0, 13.0]
    electronic_expected_list = [0.0, 4.0, 8.0, 2.0, 6.0, 4.0]

    # ionic and total only if available
    if tensor.ref.relaxed_ion is not None:
        Assert.allclose(db_dict["total_3d_tensor"], relaxed_ion_expected_list)
        Assert.allclose(db_dict["ionic_3d_tensor"], ionic_expected_list)

        assert db_dict["total_3d_isotropic_dielectric_constant"] == pytest.approx(
            float(np.trace(tensor.ref.relaxed_ion) / 3.0)
        )
        assert db_dict["ionic_3d_isotropic_dielectric_constant"] == pytest.approx(
            float(np.trace(tensor.ref.relaxed_ion - tensor.ref.clamped_ion) / 3.0)
        )

        assert db_dict["total_2d_polarizability"] == pytest.approx(
            tensor.ref.polarizability_2d
        )
        assert db_dict["ionic_2d_polarizability"] == pytest.approx(
            tensor.ref.polarizability_2d_ionic
        )
    else:
        assert db_dict["total_3d_tensor"] is None
        assert db_dict["ionic_3d_tensor"] is None
        assert db_dict["total_3d_isotropic_dielectric_constant"] is None
        assert db_dict["ionic_3d_isotropic_dielectric_constant"] is None
        assert db_dict["total_2d_polarizability"] is None
        assert db_dict["ionic_2d_polarizability"] is None

    # electronic should never be None
    Assert.allclose(db_dict["electronic_3d_tensor"], electronic_expected_list)
    assert db_dict["electronic_3d_isotropic_dielectric_constant"] == pytest.approx(
        float(np.trace(tensor.ref.clamped_ion) / 3.0)
    )
    assert db_dict["electronic_2d_polarizability"] == pytest.approx(
        tensor.ref.polarizability_2d_electronic
    )

    # check types
    for k, v in db_dict.items():
        if k.endswith("isotropic_dielectric_constant"):
            assert v is None or isinstance(v, (int, float))
        if k.endswith("polarizability"):
            assert v is None or isinstance(v, (int, float))
        elif k.endswith("tensor"):
            assert v is None or (
                isinstance(v, list) and all(isinstance(i, (int, float)) for i in v)
            )
            assert v is None or len(v) == 6
        elif k == "method":
            assert v is None or isinstance(v, str)


def test_to_database_dft(dft_tensor, Assert):
    _check_to_database(dft_tensor, Assert)


def test_to_database_rpa(rpa_tensor, Assert):
    _check_to_database(rpa_tensor, Assert)


def test_to_database_scf(scf_tensor, Assert):
    _check_to_database(scf_tensor, Assert)


def test_to_database_nscf(nscf_tensor, Assert):
    _check_to_database(nscf_tensor, Assert)


def test_to_database_slab_cell(tensor_with_slab_cell, Assert):
    _check_to_database(tensor_with_slab_cell, Assert)
