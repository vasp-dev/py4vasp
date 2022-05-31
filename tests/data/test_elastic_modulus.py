# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pytest
import types
from py4vasp.data import ElasticModulus


@pytest.fixture
def elastic_modulus(raw_data):
    raw_elastic_modulus = raw_data.elastic_modulus("dft")
    elastic_modulus = ElasticModulus.from_data(raw_elastic_modulus)
    elastic_modulus.ref = types.SimpleNamespace()
    elastic_modulus.ref.clamped_ion = raw_elastic_modulus.clamped_ion
    elastic_modulus.ref.relaxed_ion = raw_elastic_modulus.relaxed_ion
    return elastic_modulus


def test_read(elastic_modulus, Assert):
    actual = elastic_modulus.read()
    Assert.allclose(actual["clamped_ion"], elastic_modulus.ref.clamped_ion)
    Assert.allclose(actual["relaxed_ion"], elastic_modulus.ref.relaxed_ion)


def test_print(elastic_modulus, format_):
    actual, _ = format_(elastic_modulus)
    reference = f"""
Elastic modulus (kBar)
Direction    XX          YY          ZZ          XY          YZ          ZX
--------------------------------------------------------------------------------
                                  clamped-ion
XX           0.0000      4.0000      8.0000      2.0000      6.0000      4.0000
YY          36.0000     40.0000     44.0000     38.0000     42.0000     40.0000
ZZ          72.0000     76.0000     80.0000     74.0000     78.0000     76.0000
XY          18.0000     22.0000     26.0000     20.0000     24.0000     22.0000
YZ          54.0000     58.0000     62.0000     56.0000     60.0000     58.0000
ZX          36.0000     40.0000     44.0000     38.0000     42.0000     40.0000
                                  relaxed-ion
XX          81.0000     85.0000     89.0000     83.0000     87.0000     85.0000
YY         117.0000    121.0000    125.0000    119.0000    123.0000    121.0000
ZZ         153.0000    157.0000    161.0000    155.0000    159.0000    157.0000
XY          99.0000    103.0000    107.0000    101.0000    105.0000    103.0000
YZ         135.0000    139.0000    143.0000    137.0000    141.0000    139.0000
ZX         117.0000    121.0000    125.0000    119.0000    123.0000    121.0000
""".strip()
    assert actual == {"text/plain": reference}


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.elastic_modulus("dft")
    check_factory_methods(ElasticModulus, data)
