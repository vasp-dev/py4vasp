# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pytest
import types
from py4vasp.data import DielectricTensor


@pytest.fixture
def dft_tensor(raw_data):
    return make_reference(raw_data, "dft")


@pytest.fixture
def rpa_tensor(raw_data):
    return make_reference(raw_data, "rpa")


@pytest.fixture
def scf_tensor(raw_data):
    return make_reference(raw_data, "scf")


@pytest.fixture
def nscf_tensor(raw_data):
    return make_reference(raw_data, "nscf")


def make_reference(raw_data, method):
    raw_tensor = raw_data.dielectric_tensor(method)
    tensor = DielectricTensor.from_data(raw_tensor)
    tensor.ref = types.SimpleNamespace()
    tensor.ref.clamped_ion = raw_tensor.electron
    tensor.ref.relaxed_ion = raw_tensor.ion + raw_tensor.electron
    tensor.ref.independent_particle = raw_tensor.independent_particle
    tensor.ref.method = method
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


def test_print_dft_tensor(dft_tensor, format_):
    actual, _ = format_(dft_tensor)
    expected_description = "including local field effects in DFT"
    check_print_dielectric_tensor(actual, expected_description)


def test_print_rpa_tensor(rpa_tensor, format_):
    actual, _ = format_(rpa_tensor)
    expected_description = "including local field effects in RPA (Hartree)"
    check_print_dielectric_tensor(actual, expected_description)


def test_print_scf_tensor(scf_tensor, format_):
    actual, _ = format_(scf_tensor)
    expected_description = "including local field effects"
    check_print_dielectric_tensor(actual, expected_description)


def test_print_dft_tensor(nscf_tensor, format_):
    actual, _ = format_(nscf_tensor)
    expected_description = "excluding local field effects"
    check_print_dielectric_tensor(actual, expected_description)


def check_print_dielectric_tensor(actual, expected_description):
    reference = f"""
Macroscopic static dielectric tensor (dimensionless)
  {expected_description}
------------------------------------------------------
                      clamped-ion
          0.000000     1.000000     2.000000
          3.000000     4.000000     5.000000
          6.000000     7.000000     8.000000
                      relaxed-ion
          9.000000    11.000000    13.000000
         15.000000    17.000000    19.000000
         21.000000    23.000000    25.000000
""".strip()
    assert actual == {"text/plain": reference}


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.dielectric_tensor("dft")
    check_factory_methods(DielectricTensor, data)
