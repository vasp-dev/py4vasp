# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pytest

from py4vasp import calculation, exception


def test_access_of_attributes():
    for key in calculation.__all__:
        getattr(calculation, key)
    calculation.dos  # access one random quantity to make sure that __all__ includes some keys


def test_nonexisting_attribute():
    with pytest.raises(exception.MissingAttribute):
        calculation.does_not_exist
