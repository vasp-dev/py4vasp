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
    tensor = DielectricTensor(raw_tensor)
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
    actual = dielectric_tensor.read()
    Assert.allclose(actual["clamped_ion"], dielectric_tensor.ref.clamped_ion)
    Assert.allclose(actual["relaxed_ion"], dielectric_tensor.ref.relaxed_ion)
    Assert.allclose(
        actual["independent_particle"], dielectric_tensor.ref.independent_particle
    )
    assert actual["method"] == dielectric_tensor.ref.method


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
MACROSCOPIC STATIC DIELECTRIC TENSOR ({expected_description})
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


def test_descriptor(dft_tensor, check_descriptors):
    descriptors = {
        "_to_dict": ["to_dict", "read"],
        "_to_string": ["__str__"],
    }
    check_descriptors(dft_tensor, descriptors)


def test_from_file(raw_data, mock_file, check_read):
    raw_dielectric_tensor = raw_data.dielectric_tensor("dft")
    with mock_file("dielectric_tensor", raw_dielectric_tensor) as mocks:
        check_read(DielectricTensor, mocks, raw_dielectric_tensor)
