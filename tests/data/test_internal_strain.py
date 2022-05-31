# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pytest
import types
from py4vasp.data import InternalStrain, Structure


@pytest.fixture
def Sr2TiO4(raw_data):
    raw_internal_strain = raw_data.internal_strain("Sr2TiO4")
    internal_strain = InternalStrain.from_data(raw_internal_strain)
    internal_strain.ref = types.SimpleNamespace()
    internal_strain.ref.structure = Structure(raw_internal_strain.structure)
    internal_strain.ref.internal_strain = raw_internal_strain.internal_strain
    return internal_strain


def test_Sr2TiO4_read(Sr2TiO4, Assert):
    actual = Sr2TiO4.read()
    reference_structure = Sr2TiO4.ref.structure.read()
    for key in actual["structure"]:
        if key in ("elements", "names"):
            assert actual["structure"][key] == reference_structure[key]
        else:
            Assert.allclose(actual["structure"][key], reference_structure[key])
    Assert.allclose(actual["internal_strain"], Sr2TiO4.ref.internal_strain)


def test_Sr2TiO4_print(Sr2TiO4, format_):
    actual, _ = format_(Sr2TiO4)
    reference = """
Internal strain tensor (eV/Å):
 ion  displ     X           Y           Z          XY          YZ          ZX
---------------------------------------------------------------------------------
   1    x     0.00000     4.00000     8.00000     2.00000     6.00000     4.00000
        y     9.00000    13.00000    17.00000    11.00000    15.00000    13.00000
        z    18.00000    22.00000    26.00000    20.00000    24.00000    22.00000
   2    x    27.00000    31.00000    35.00000    29.00000    33.00000    31.00000
        y    36.00000    40.00000    44.00000    38.00000    42.00000    40.00000
        z    45.00000    49.00000    53.00000    47.00000    51.00000    49.00000
   3    x    54.00000    58.00000    62.00000    56.00000    60.00000    58.00000
        y    63.00000    67.00000    71.00000    65.00000    69.00000    67.00000
        z    72.00000    76.00000    80.00000    74.00000    78.00000    76.00000
   4    x    81.00000    85.00000    89.00000    83.00000    87.00000    85.00000
        y    90.00000    94.00000    98.00000    92.00000    96.00000    94.00000
        z    99.00000   103.00000   107.00000   101.00000   105.00000   103.00000
   5    x   108.00000   112.00000   116.00000   110.00000   114.00000   112.00000
        y   117.00000   121.00000   125.00000   119.00000   123.00000   121.00000
        z   126.00000   130.00000   134.00000   128.00000   132.00000   130.00000
   6    x   135.00000   139.00000   143.00000   137.00000   141.00000   139.00000
        y   144.00000   148.00000   152.00000   146.00000   150.00000   148.00000
        z   153.00000   157.00000   161.00000   155.00000   159.00000   157.00000
   7    x   162.00000   166.00000   170.00000   164.00000   168.00000   166.00000
        y   171.00000   175.00000   179.00000   173.00000   177.00000   175.00000
        z   180.00000   184.00000   188.00000   182.00000   186.00000   184.00000
""".strip()
    assert actual == {"text/plain": reference}


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.internal_strain("Sr2TiO4")
    check_factory_methods(InternalStrain, data)
