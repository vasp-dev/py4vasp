# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pytest

from py4vasp import Calculation, calculation, exception


def test_access_of_attributes():
    calc = Calculation.from_path(".")
    for key in filter(attribute_included, dir(calc)):
        getattr(calculation, key)


def attribute_included(attr):
    if attr.startswith("_"):  # do not include private attributes
        return False
    if attr.startswith("from"):  # do not include classmethods
        return False
    return True


def test_nonexisting_attribute():
    with pytest.raises(exception.MissingAttribute):
        calculation.does_not_exist
