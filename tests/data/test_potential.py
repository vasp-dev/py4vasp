# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import pytest

from py4vasp.data import Potential, Structure


@pytest.fixture
def reference_potential(raw_data):
    def _reference_potential(
        selection, hartree_potential, ionic_potential, xc_potential
    ):
        raw_potential = raw_data.potential(
            selection=selection,
            hartree_potential=hartree_potential,
            ionic_potential=ionic_potential,
            xc_potential=xc_potential,
        )
        potential = Potential.from_data(raw_potential)
        potential.ref = types.SimpleNamespace()
        potential.ref.structure = Structure.from_data(raw_potential.structure).read()
        potential.ref.total_potential = raw_potential.total_potential
        if hartree_potential:
            potential.ref.hartree_potential = raw_potential.hartree_potential
        if ionic_potential:
            potential.ref.ionic_potential = raw_potential.ionic_potential
        if xc_potential:
            potential.ref.xc_potential = raw_potential.xc_potential
        return potential

    return _reference_potential


def separate_potentials(selection, potential_name, potential):
    output = {}
    _potential = getattr(potential.ref, f"{potential_name}_potential")
    _potential = _potential.__array__()
    if selection == "non_spin_polarized":
        output[f"{potential_name}"] = _potential[0]
    elif selection == "collinear":
        output[f"{potential_name}"] = (_potential[0] + _potential[1]) / 2
        output[f"{potential_name}_up"] = _potential[0]
        output[f"{potential_name}_down"] = _potential[1]
    elif selection == "non_collinear":
        output[f"{potential_name}"] = _potential[0]
        output[f"{potential_name}_magnetization"] = _potential[1:4]
    return output


@pytest.fixture
def potential_data(reference_potential):
    def _potential_data(
        selection: str,
        hartree_potential: bool,
        ionic_potential: bool,
        xc_potential: bool,
    ):
        reference = reference_potential(
            selection, hartree_potential, ionic_potential, xc_potential
        )
        total_potential_data = separate_potentials(selection, "total", reference)
        if hartree_potential:
            hartree_potential_data = separate_potentials(
                selection, "hartree", reference
            )
            total_potential_data.update(hartree_potential_data)
        if ionic_potential:
            ionic_potential_data = separate_potentials(selection, "ionic", reference)
            total_potential_data.update(ionic_potential_data)
        if xc_potential:
            xc_potential_data = separate_potentials(selection, "xc", reference)
            total_potential_data.update(xc_potential_data)
        reference.ref.to_dict = total_potential_data
        return reference

    return _potential_data


@pytest.mark.parametrize(
    "selection", ["non_spin_polarized", "collinear", "non_collinear"]
)
@pytest.mark.parametrize("hartree_potential", [True, False])
@pytest.mark.parametrize("ionic_potential", [True, False])
@pytest.mark.parametrize("xc_potential", [True, False])
def test_read(
    potential_data, selection, hartree_potential, ionic_potential, xc_potential, Assert
):
    potential = potential_data(
        selection, hartree_potential, ionic_potential, xc_potential
    )
    output_potential = potential.read()
    output_keys = list(output_potential.keys())
    expected_potential = potential.ref.to_dict
    for key in output_keys:
        _output = output_potential.pop(key)
        _expected = expected_potential.pop(key)
        Assert.allclose(_output, _expected)
    assert not output_potential  # Must be empty
    assert not expected_potential  # Must be empty
