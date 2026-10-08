# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types
from dataclasses import fields

import numpy as np
import pytest

from py4vasp._calculation.born_effective_charge import (
    BornEffectiveCharge,
    BornEffectiveChargeHandler,
)
from py4vasp._calculation.structure import StructureHandler
from py4vasp._raw.models import BornEffectiveChargeModel


@pytest.fixture
def Sr2TiO4(raw_data):
    raw_born_charges = raw_data.born_effective_charge("Sr2TiO4")
    handler = BornEffectiveChargeHandler.from_data(raw_born_charges)
    handler.ref = types.SimpleNamespace()
    structure = StructureHandler.from_data(raw_born_charges.structure)
    handler.ref.structure = structure
    handler.ref.charge_tensors = raw_born_charges.charge_tensors
    handler.ref.minmax_info = (12, 0, 174, 6)
    handler.ref.raw_data = raw_born_charges
    return handler


@pytest.fixture
def dispatcher(raw_data):
    return BornEffectiveCharge.from_data(raw_data.born_effective_charge("Sr2TiO4"))


def test_Sr2TiO4_read(Sr2TiO4, Assert):
    actual = Sr2TiO4.read()
    reference_structure = Sr2TiO4.ref.structure.to_dict()
    for key in actual["structure"]:
        if key in ("elements", "names"):
            assert actual["structure"][key] == reference_structure[key]
        else:
            Assert.allclose(actual["structure"][key], reference_structure[key])
    Assert.allclose(actual["charge_tensors"], Sr2TiO4.ref.charge_tensors)


REFERENCE_OUTPUT = """
BORN EFFECTIVE CHARGES (including local field effects) (in |e|, cumulative output)
---------------------------------------------------------------------------------
ion    1   Sr
    1     0.00000     3.00000     6.00000
    2     1.00000     4.00000     7.00000
    3     2.00000     5.00000     8.00000
ion    2   Sr
    1     9.00000    12.00000    15.00000
    2    10.00000    13.00000    16.00000
    3    11.00000    14.00000    17.00000
ion    3   Ti
    1    18.00000    21.00000    24.00000
    2    19.00000    22.00000    25.00000
    3    20.00000    23.00000    26.00000
ion    4   O
    1    27.00000    30.00000    33.00000
    2    28.00000    31.00000    34.00000
    3    29.00000    32.00000    35.00000
ion    5   O
    1    36.00000    39.00000    42.00000
    2    37.00000    40.00000    43.00000
    3    38.00000    41.00000    44.00000
ion    6   O
    1    45.00000    48.00000    51.00000
    2    46.00000    49.00000    52.00000
    3    47.00000    50.00000    53.00000
ion    7   O
    1    54.00000    57.00000    60.00000
    2    55.00000    58.00000    61.00000
    3    56.00000    59.00000    62.00000
""".strip()


def test_Sr2TiO4_print(Sr2TiO4, format_):
    actual, _ = format_(Sr2TiO4)
    assert actual == {"text/plain": REFERENCE_OUTPUT}


def test_print_dispatcher(dispatcher, format_):
    actual, _ = format_(dispatcher)
    assert actual == {"text/plain": REFERENCE_OUTPUT}


def test_print_writes_to_stdout(dispatcher, capsys):
    assert dispatcher.print() is None
    assert capsys.readouterr().out == str(dispatcher) + "\n"


def test_selections(dispatcher):
    assert dispatcher.selections() == {"born_effective_charge": ["default"]}


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.born_effective_charge("Sr2TiO4")
    check_factory_methods(BornEffectiveCharge, data, skip_methods=["selections"])


def test_to_database(Sr2TiO4):
    born_db: BornEffectiveChargeModel = Sr2TiO4.to_database()
    assert born_db.eigenvalue_min == Sr2TiO4.ref.minmax_info[0]
    assert born_db.eigenvalue_max == Sr2TiO4.ref.minmax_info[2]
    assert born_db.eigenvalue_min_index == Sr2TiO4.ref.minmax_info[1]
    assert born_db.eigenvalue_max_index == Sr2TiO4.ref.minmax_info[3]

    for fld in fields(BornEffectiveChargeModel):
        if fld.name.startswith("__"):
            assert isinstance(getattr(born_db, fld.name), str)
        elif fld.name.endswith("index"):
            assert getattr(born_db, fld.name) is None or isinstance(
                getattr(born_db, fld.name), int
            )
        else:
            assert getattr(born_db, fld.name) is None or isinstance(
                getattr(born_db, fld.name), float
            )


def test_Sr2TiO4_to_INCAR(Sr2TiO4):
    expected = """\
PHON_BORN_CHARGES =   0.000000   3.000000   6.000000   1.000000   4.000000   7.000000   2.000000   5.000000   8.000000 \\
                      9.000000  12.000000  15.000000  10.000000  13.000000  16.000000  11.000000  14.000000  17.000000 \\
                     18.000000  21.000000  24.000000  19.000000  22.000000  25.000000  20.000000  23.000000  26.000000 \\
                     27.000000  30.000000  33.000000  28.000000  31.000000  34.000000  29.000000  32.000000  35.000000 \\
                     36.000000  39.000000  42.000000  37.000000  40.000000  43.000000  38.000000  41.000000  44.000000 \\
                     45.000000  48.000000  51.000000  46.000000  49.000000  52.000000  47.000000  50.000000  53.000000 \\
                     54.000000  57.000000  60.000000  55.000000  58.000000  61.000000  56.000000  59.000000  62.000000
"""
    assert Sr2TiO4.to_INCAR() == expected


@pytest.mark.parametrize("selection", (None, "default"))
def test_to_INCAR_dispatcher(dispatcher, Sr2TiO4, selection):
    assert dispatcher.to_INCAR(selection) == Sr2TiO4.to_INCAR()


def test_Sr2TiO4_to_INCAR_orientation(Sr2TiO4, Assert):
    # VASP reads the list into BORN_EFF_CHARGES(3,3,NIONS) in Fortran order and then
    # transposes every 3x3 block (phonon.F). The first index is the field direction
    # (polar_ewald.F contracts it with q). VASP writes the Born charges to vaspout.h5
    # with the field as the last numpy axis, so the INCAR holds every block transposed.
    _, values = Sr2TiO4.to_INCAR().split("=")
    values = np.array(values.replace("\\", " ").split(), dtype=float)
    number_ions = len(Sr2TiO4.ref.charge_tensors)
    charges_in_vasp = values.reshape(number_ions, 3, 3)
    Assert.allclose(charges_in_vasp, np.swapaxes(Sr2TiO4.ref.charge_tensors, 1, 2))


def test_print_rows_match_to_INCAR(Sr2TiO4):
    printed_rows = [
        line.split()[1:]
        for line in str(Sr2TiO4).splitlines()
        if line.startswith("    ")
    ]
    printed = np.array(printed_rows, dtype=float).reshape(-1, 9)
    tag_values = Sr2TiO4.to_INCAR().replace("PHON_BORN_CHARGES =", "")
    incar = np.array(tag_values.replace("\\", "").split(), dtype=float)
    assert np.array_equal(printed, incar.reshape(-1, 9))
