# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses
import pytest
from unittest.mock import patch, PropertyMock
from py4vasp.raw import File, RawVersion
import py4vasp.exceptions as exception


def test_vasp_6_3_features():
    new_properties = (
        "born_effective_charge",
        "dielectric_function",
        "dielectric_tensor",
        "elastic_modulus",
        "force_constant",
        "internal_strain",
        "piezoelectric_tensor",
        "polarization",
    )
    for property_ in new_properties:
        check_property_requires_version(property_, RawVersion(6, 3))


def check_property_requires_version(property_, required_version):
    outdated_patch = required_version.patch - 1
    outdated_version = dataclasses.replace(required_version, patch=outdated_patch)
    access_property(property_, required_version)  # no exception
    with pytest.raises(exception.OutdatedVaspVersion):
        access_property(property_, outdated_version)


def access_property(property_, vasp_version):
    version_property = PropertyMock(return_value=vasp_version)
    cm_init = patch.object(File, "__init__", return_value=None)
    cm_version = patch.object(File, "version", new_callable=version_property)
    cm_inner_func = patch.object(File, f"_read_{property_}")
    with cm_init, cm_version, cm_inner_func:
        file_ = File()
        getattr(file_, property_)
