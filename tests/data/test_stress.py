# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest
import types

from py4vasp.data import Stress, Structure
import py4vasp.exceptions as exception


@pytest.fixture
def Sr2TiO4(raw_data):
    raw_stress = raw_data.stress("Sr2TiO4")
    stress = Stress(raw_stress)
    stress.ref = types.SimpleNamespace()
    stress.ref.structure = Structure(raw_stress.structure)
    stress.ref.stress = raw_stress.stress
    return stress


@pytest.fixture
def Fe3O4(raw_data):
    raw_stress = raw_data.stress("Fe3O4")
    stress = Stress(raw_stress)
    stress.ref = types.SimpleNamespace()
    stress.ref.structure = Structure(raw_stress.structure)
    stress.ref.stress = raw_stress.stress
    return stress


def test_read_Sr2TiO4(Sr2TiO4, Assert):
    check_read_stress(Sr2TiO4.read(), Sr2TiO4.ref, -1, Assert)
    for steps in (slice(None), slice(1, 3), 0):
        check_read_stress(Sr2TiO4[steps].read(), Sr2TiO4.ref, steps, Assert)


def test_read_Fe3O4(Fe3O4, Assert):
    check_read_stress(Fe3O4.read(), Fe3O4.ref, -1, Assert)
    for steps in (slice(None), slice(1, 3), 0):
        check_read_stress(Fe3O4[steps].read(), Fe3O4.ref, steps, Assert)


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


def test_descriptor(Sr2TiO4, check_descriptors):
    descriptors = {
        "_to_dict": ["to_dict", "read"],
        "_to_string": ["__str__"],
    }
    check_descriptors(Sr2TiO4, descriptors)


def test_from_file(raw_data, mock_file, check_read):
    raw_stress = raw_data.stress("Sr2TiO4")
    with mock_file("stress", raw_stress) as mocks:
        check_read(Stress, mocks, raw_stress)
