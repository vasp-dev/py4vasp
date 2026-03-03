# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import pytest

from py4vasp import exception
from py4vasp._calculation.stress import Stress
from py4vasp._calculation.structure import Structure
from py4vasp._raw.data_db import Stress_DB


@pytest.fixture
def Sr2TiO4(raw_data):
    raw_stress = raw_data.stress("Sr2TiO4")
    stress = Stress.from_data(raw_stress)
    stress.ref = types.SimpleNamespace()
    stress.ref.structure = Structure.from_data(raw_stress.structure)
    stress.ref.stress = raw_stress.stress
    return stress


@pytest.fixture
def Fe3O4(raw_data):
    raw_stress = raw_data.stress("Fe3O4")
    stress = Stress.from_data(raw_stress)
    stress.ref = types.SimpleNamespace()
    stress.ref.structure = Structure.from_data(raw_stress.structure)
    stress.ref.stress = raw_stress.stress
    return stress


@pytest.fixture(params=["Sr2TiO4", "Fe3O4"])
def stresses(request, raw_data):
    raw_stress = raw_data.stress(request.param)
    stress = Stress.from_data(raw_stress)
    stress.ref = types.SimpleNamespace()
    stress.ref.structure = Structure.from_data(raw_stress.structure)
    stress.ref.stress = raw_stress.stress
    return stress


def test_read(stresses, Assert):
    check_read_stress(stresses.read(), stresses.ref, -1, Assert)
    for steps in (slice(None), slice(1, 3), 0):
        check_read_stress(stresses[steps].read(), stresses.ref, steps, Assert)


def check_read_stress(actual, reference, steps, Assert):
    reference_structure = reference.structure[steps].read()
    for key in actual["structure"]:
        if key in ("elements", "names"):
            assert actual["structure"][key] == reference_structure[key]
        else:
            Assert.allclose(actual["structure"][key], reference_structure[key])
    Assert.allclose(actual["stress"], reference.stress[steps])


def test_incorrect_access(Sr2TiO4):
    out_of_bounds = 999
    with pytest.raises(exception.IncorrectUsage):
        Sr2TiO4[out_of_bounds].read()
    with pytest.raises(exception.IncorrectUsage):
        Sr2TiO4["string instead of int"].read()


def test_print_Sr2TiO4(Sr2TiO4, format_):
    actual, _ = format_(Sr2TiO4)
    ref_plain = """
FORCE on cell =-STRESS in cart. coord.  units (eV):
Direction    XX          YY          ZZ          XY          YZ          ZX
-------------------------------------------------------------------------------------
Total       1.64862     1.89286     2.13710     1.77074     2.01498     1.89286
in kB      27.00000    31.00000    35.00000    29.00000    33.00000    31.00000
""".strip()
    assert actual == {"text/plain": ref_plain}
    #
    actual, _ = format_(Sr2TiO4[0])
    ref_plain = """
FORCE on cell =-STRESS in cart. coord.  units (eV):
Direction    XX          YY          ZZ          XY          YZ          ZX
-------------------------------------------------------------------------------------
Total       0.00000     0.24424     0.48848     0.12212     0.36636     0.24424
in kB       0.00000     4.00000     8.00000     2.00000     6.00000     4.00000
""".strip()
    assert actual == {"text/plain": ref_plain}
    #
    actual, _ = format_(Sr2TiO4[1:3])
    ref_plain = """
FORCE on cell =-STRESS in cart. coord.  units (eV):
Direction    XX          YY          ZZ          XY          YZ          ZX
-------------------------------------------------------------------------------------
Total       1.09908     1.34332     1.58756     1.22120     1.46544     1.34332
in kB      18.00000    22.00000    26.00000    20.00000    24.00000    22.00000
""".strip()
    assert actual == {"text/plain": ref_plain}


def test_to_database(stresses, Assert):
    db_data: Stress_DB = stresses._read_to_database()["stress:default"]
    assert isinstance(db_data, Stress_DB)
    initial_tensor = stresses.ref.stress[0]
    final_tensor = stresses.ref.stress[-1]

    assert db_data.initial_stress_mean == pytest.approx(
        (initial_tensor[0, 0] + initial_tensor[1, 1] + initial_tensor[2, 2]) / 3.0
    )
    assert db_data.final_stress_mean == pytest.approx(
        (final_tensor[0, 0] + final_tensor[1, 1] + final_tensor[2, 2]) / 3.0
    )

    reduced_final_tensor = [
        final_tensor[0, 0],
        final_tensor[1, 1],
        final_tensor[2, 2],
        0.5 * (final_tensor[0, 1] + final_tensor[1, 0]),
        0.5 * (final_tensor[1, 2] + final_tensor[2, 1]),
        0.5 * (final_tensor[0, 2] + final_tensor[2, 0]),
    ]

    Assert.allclose(db_data.final_stress_tensor, reduced_final_tensor)


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.stress("Sr2TiO4")
    check_factory_methods(Stress, data)
