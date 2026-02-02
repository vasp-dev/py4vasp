# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import numpy as np
import pytest

from py4vasp._calculation.elastic_modulus import (
    ElasticModulus,
    _ElasticTensor,
    _get_C_voigt_from_4d_tensor,
)
from py4vasp._util.tensor import symmetry_reduce


@pytest.fixture
def elastic_modulus(raw_data):
    return _setup_elastic_modulus(raw_data, "dft")


@pytest.fixture(params=["dft", "dft with structure", "SiC"])
def elastic_moduli(raw_data, request):
    return _setup_elastic_modulus(raw_data, request.param)


def _setup_elastic_modulus(raw_data, selection):
    raw_elastic_modulus = raw_data.elastic_modulus(selection)
    elastic_modulus = ElasticModulus.from_data(raw_elastic_modulus)
    elastic_modulus.ref = types.SimpleNamespace()
    elastic_modulus.ref.structure = raw_elastic_modulus.structure
    elastic_modulus.ref.clamped_ion = raw_elastic_modulus.clamped_ion
    elastic_modulus.ref.relaxed_ion = raw_elastic_modulus.relaxed_ion
    elastic_modulus.ref.overview_data = _setup_overview_data(elastic_modulus)
    if selection == "SiC":
        elastic_modulus.ref.overview_data["bulk_modulus"] = 227.9149991624522
        elastic_modulus.ref.overview_data["shear_modulus"] = 197.09002974236228
        elastic_modulus.ref.overview_data["youngs_modulus"] = 458.9712638295085
        elastic_modulus.ref.overview_data["poisson_ratio"] = 0.16436956356818136
        elastic_modulus.ref.overview_data["pugh_ratio"] = 0.8647523439292442
        elastic_modulus.ref.overview_data["vickers_hardness"] = 31.07791529386013
        elastic_modulus.ref.overview_data["fracture_toughness"] = 3.3895949025923313
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


def test_to_database(elastic_moduli):
    # TODO improve test with actual numbers or write unit tests for _ElasticTensor
    database_data = elastic_moduli._read_to_database()
    overview = database_data["elastic_modulus:default"]
    ref_overview = elastic_moduli.ref.overview_data
    for key, value in ref_overview.items():
        if (key == "fracture_toughness") and (elastic_moduli.ref.structure is None):
            assert (
                overview[key] is None
            ), f"fracture_toughness requires structure data: but returned db value is {overview[key]}."
        elif value is None:
            if (
                np.abs(
                    np.linalg.det(
                        _get_C_voigt_from_4d_tensor(elastic_moduli.ref.relaxed_ion)
                    )
                )
                > 1e-14
            ):
                assert (
                    overview[key] is not None
                ), f"expected non-None value for {key}, but got {overview[key]}."
        else:
            assert np.all(
                np.array(np.isclose(overview[key], value))
            ), f"mismatch in {key}: expected {value}, got {overview[key]}."


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.elastic_modulus("dft")
    check_factory_methods(ElasticModulus, data)
