# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pytest
import types
from py4vasp.data import BornEffectiveCharge, Structure


@pytest.fixture
def Sr2TiO4(raw_data):
    raw_born_charges = raw_data.born_effective_charge("Sr2TiO4")
    born_charges = BornEffectiveCharge(raw_born_charges)
    born_charges.ref = types.SimpleNamespace()
    born_charges.ref.structure = Structure(raw_born_charges.structure)
    born_charges.ref.charge_tensors = raw_born_charges.charge_tensors
    return born_charges


def test_Sr2TiO4_read(Sr2TiO4, Assert):
    actual = Sr2TiO4.read()
    reference_structure = Sr2TiO4.ref.structure.read()
    for key in actual["structure"]:
        if key in ("elements", "names"):
            assert actual["structure"][key] == reference_structure[key]
        else:
            Assert.allclose(actual["structure"][key], reference_structure[key])
    Assert.allclose(actual["charge_tensors"], Sr2TiO4.ref.charge_tensors)


def test_Sr2TiO4_print(Sr2TiO4, format_):
    actual, _ = format_(Sr2TiO4)
    reference = """
BORN EFFECTIVE CHARGES (including local field effects) (in |e|, cumulative output)
---------------------------------------------------------------------------------
ion    1   Sr
    1     0.00000     1.00000     2.00000
    2     3.00000     4.00000     5.00000
    3     6.00000     7.00000     8.00000
ion    2   Sr
    1     9.00000    10.00000    11.00000
    2    12.00000    13.00000    14.00000
    3    15.00000    16.00000    17.00000
ion    3   Ti
    1    18.00000    19.00000    20.00000
    2    21.00000    22.00000    23.00000
    3    24.00000    25.00000    26.00000
ion    4   O
    1    27.00000    28.00000    29.00000
    2    30.00000    31.00000    32.00000
    3    33.00000    34.00000    35.00000
ion    5   O
    1    36.00000    37.00000    38.00000
    2    39.00000    40.00000    41.00000
    3    42.00000    43.00000    44.00000
ion    6   O
    1    45.00000    46.00000    47.00000
    2    48.00000    49.00000    50.00000
    3    51.00000    52.00000    53.00000
ion    7   O
    1    54.00000    55.00000    56.00000
    2    57.00000    58.00000    59.00000
    3    60.00000    61.00000    62.00000
""".strip()
    assert actual == {"text/plain": reference}


def test_descriptor(Sr2TiO4, check_descriptors):
    descriptors = {
        "_to_dict": ["to_dict", "read"],
        "_to_string": ["__str__"],
    }
    check_descriptors(Sr2TiO4, descriptors)


def test_from_file(raw_data, mock_file, check_read):
    raw_born_charges = raw_data.born_effective_charge("Sr2TiO4")
    with mock_file("born_effective_charge", raw_born_charges) as mocks:
        check_read(BornEffectiveCharge, mocks, raw_born_charges)
