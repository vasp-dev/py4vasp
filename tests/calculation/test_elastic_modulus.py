# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import pytest

from py4vasp._calculation.elastic_modulus import ElasticModulus, _ElasticTensor
from py4vasp._util.tensor import symmetry_reduce


@pytest.fixture
def elastic_modulus(raw_data):
    raw_elastic_modulus = raw_data.elastic_modulus("dft")
    elastic_modulus = ElasticModulus.from_data(raw_elastic_modulus)
    elastic_modulus.ref = types.SimpleNamespace()
    elastic_modulus.ref.clamped_ion = raw_elastic_modulus.clamped_ion
    elastic_modulus.ref.relaxed_ion = raw_elastic_modulus.relaxed_ion
    elastic_modulus.ref.overview_data = _setup_overview_data(elastic_modulus)
    return elastic_modulus


def _setup_overview_data(modulus_obj):
    tensor = modulus_obj.ref.relaxed_ion
    compact_tensor = symmetry_reduce(symmetry_reduce(tensor).T).T
    return {
        "elastic_modulus_reduced": list([list(l) for l in compact_tensor]),
        "bulk_modulus": None,
        "shear_modulus": None,
        "youngs_modulus": None,
        "poisson_ratio": None,
        "pugh_ratio": None,
        "vickers_hardness": None,
        "fracture_toughness": None,
    }


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
XX           1.0000      5.0000      9.0000      3.0000      7.0000      5.0000
YY          37.0000     41.0000     45.0000     39.0000     43.0000     41.0000
ZZ          73.0000     77.0000     81.0000     75.0000     79.0000     77.0000
XY          19.0000     23.0000     27.0000     21.0000     25.0000     23.0000
YZ          55.0000     59.0000     63.0000     57.0000     61.0000     59.0000
ZX          37.0000     41.0000     45.0000     39.0000     43.0000     41.0000
                                  relaxed-ion
XX          82.0000     86.0000     90.0000     84.0000     88.0000     86.0000
YY         118.0000    122.0000    126.0000    120.0000    124.0000    122.0000
ZZ         154.0000    158.0000    162.0000    156.0000    160.0000    158.0000
XY         100.0000    104.0000    108.0000    102.0000    106.0000    104.0000
YZ         136.0000    140.0000    144.0000    138.0000    142.0000    140.0000
ZX         118.0000    122.0000    126.0000    120.0000    124.0000    122.0000
""".strip()
    assert actual == {"text/plain": reference}


def test_to_database(elastic_modulus):
    database_data = elastic_modulus._to_database()
    overview = database_data["elastic_modulus"]
    ref_overview = elastic_modulus.ref.overview_data
    for key, value in ref_overview.items():
        if value is None:
            assert overview[key] is not None or (
                key == "fracture_toughness"
                and elastic_modulus._raw_data.structure is None
            ), f"{key} is None in database output: {overview}"
        else:
            assert overview[key] == value


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.elastic_modulus("dft")
    check_factory_methods(ElasticModulus, data)
