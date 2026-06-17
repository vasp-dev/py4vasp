# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pytest

from py4vasp import Calculation, calculation, demo


def test_access_of_attributes():
    calc = Calculation.from_path(".")
    for key in filter(attribute_included, dir(calc)):
        getattr(calculation, key)


def attribute_included(attr):
    if attr.startswith("_"):  # do not include private attributes
        return False
    if attr.startswith("from"):  # do not include classmethods
        return False
    return True


@pytest.mark.skip("Input files are not included in current release.")
def test_assigning_to_input_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    expected = "SYSTEM = demo INCAR file"
    calculation.INCAR = expected
    with open("INCAR", "r") as file:
        actual = file.read()
    assert actual == expected


def test_selections_on_empty_path(tmp_path):
    calc = Calculation.from_path(tmp_path)
    assert calc.selections() == {}


def test_selections_on_demo_calculation(tmp_path):
    calc = demo.calculation(tmp_path / "demo_calculation")
    actual = calc.selections()
    # quantities with data should expose their loadable selections
    expected = {
        "band": ["default", "kpoints_opt"],
        "current_density": ["nmr"],
        "dos": ["default", "kpoints_opt"],
        "energy": ["default"],
        "exciton.density": ["default"],
        "force": ["default"],
        "nics": ["default"],
        "partial_density": ["default"],
        "potential": ["default"],
        "run_info": ["default"],
        "stress": ["default"],
        "system": ["default"],
        "velocity": ["default"],
    }
    for quantity, selections in expected.items():
        assert actual[quantity] == selections
    # density is written to the wavefunction file with an additional kinetic part
    assert actual["density"] == ["default", "tau"]
    # structure can be read with the default and the exciton-relaxed positions
    assert actual["structure"] == ["default", "exciton"]
    # the result is sorted by quantity name
    assert list(actual) == sorted(actual)


def test_selections_excludes_quantities_without_data(tmp_path):
    calc = demo.calculation(tmp_path / "demo_calculation")
    actual = calc.selections()
    absent = (
        "bandgap",
        "born_effective_charge",
        "dielectric_function",
        "dielectric_tensor",
        "elastic_modulus",
        "internal_strain",
        "piezoelectric_tensor",
        "polarization",
    )
    for quantity in absent:
        assert quantity not in actual
