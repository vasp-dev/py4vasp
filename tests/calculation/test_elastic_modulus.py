# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import numpy as np
import pytest

from py4vasp._calculation.elastic_modulus import ElasticModulus
from py4vasp._raw.data_db import ElasticModulus_DB
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
        elastic_modulus.ref.overview_data["total_bulk_modulus"] = 227.9149991624522
        elastic_modulus.ref.overview_data["total_shear_modulus"] = 197.09002974236228
        elastic_modulus.ref.overview_data["total_young_modulus"] = 458.9712638295085
        elastic_modulus.ref.overview_data["total_poisson_ratio"] = 0.16436956356818136
        elastic_modulus.ref.overview_data["total_pugh_ratio"] = 0.8647523439292442
        elastic_modulus.ref.overview_data["total_vickers_hardness"] = 31.07791529386013
        elastic_modulus.ref.overview_data["total_fracture_toughness"] = (
            3.3895949025923313
        )
        elastic_modulus.ref.overview_data["ionic_bulk_modulus"] = -0.0001396466
        elastic_modulus.ref.overview_data["ionic_shear_modulus"] = -7.263256685782666
        elastic_modulus.ref.overview_data["ionic_young_modulus"] = -0.001256747258012485
        elastic_modulus.ref.overview_data["ionic_poisson_ratio"] = -0.9999134596733735
        elastic_modulus.ref.overview_data["ionic_pugh_ratio"] = 52011.67987366603
        elastic_modulus.ref.overview_data["ionic_vickers_hardness"] = (
            -0.003173549206852492
        )
        elastic_modulus.ref.overview_data["ionic_fracture_toughness"] = None
        elastic_modulus.ref.overview_data["electronic_bulk_modulus"] = (
            227.91517068333098
        )
        elastic_modulus.ref.overview_data["electronic_shear_modulus"] = (
            208.30942571448918
        )
        elastic_modulus.ref.overview_data["electronic_young_modulus"] = (
            478.99729799241635
        )
        elastic_modulus.ref.overview_data["electronic_poisson_ratio"] = (
            0.14972545373183094
        )
        elastic_modulus.ref.overview_data["electronic_pugh_ratio"] = 0.9139778852365982
        elastic_modulus.ref.overview_data["electronic_vickers_hardness"] = (
            34.96010439285016
        )
        elastic_modulus.ref.overview_data["electronic_fracture_toughness"] = (
            3.2513198905495826
        )
    return elastic_modulus


def _setup_overview_data(modulus_obj):
    total_tensor = modulus_obj.ref.relaxed_ion
    compact_total_tensor = symmetry_reduce(symmetry_reduce(total_tensor).T).T
    ionic_tensor = modulus_obj.ref.relaxed_ion - modulus_obj.ref.clamped_ion
    compact_ionic_tensor = symmetry_reduce(symmetry_reduce(ionic_tensor).T).T
    electronic_tensor = modulus_obj.ref.clamped_ion
    compact_electronic_tensor = symmetry_reduce(symmetry_reduce(electronic_tensor).T).T
    return_dict = {
        "total_3d_tensor": list([list(l) for l in compact_total_tensor]),
        "ionic_3d_tensor": list([list(l) for l in compact_ionic_tensor]),
        "electronic_3d_tensor": list([list(l) for l in compact_electronic_tensor]),
    }
    for primary_key in ["total", "ionic", "electronic"]:
        for secondary_key in [
            "bulk_modulus",
            "shear_modulus",
            "youngs_modulus",
            "poisson_ratio",
            "pugh_ratio",
            "vickers_hardness",
            "fracture_toughness",
        ]:
            return_dict[f"{primary_key}_{secondary_key}"] = None
    return return_dict


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
    database_data = elastic_moduli._read_to_database()
    overview: ElasticModulus_DB = database_data["elastic_modulus:default"]
    ref_overview = elastic_moduli.ref.overview_data

    for key, value in ref_overview.items():
        if (key.endswith("fracture_toughness")) and (
            elastic_moduli.ref.structure is None
        ):
            assert (
                getattr(overview, key) is None
            ), f"fracture_toughness requires structure data: but returned db value is {getattr(overview, key)}."
        elif value is None:
            continue
            # if matrix is close to singular, some properties can probably not be computed
            # in that case, skip assertion -- np.linalg.inv may or may not throw an error depending on system
        else:
            assert np.all(
                np.array(np.isclose(getattr(overview, key), value))
            ), f"mismatch in {key}: expected {value}, got {getattr(overview, key)}."


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.elastic_modulus("dft")
    check_factory_methods(ElasticModulus, data)
