import pytest

from py4vasp import exception
from py4vasp._util import import_


def test_import_not_available():
    module = import_.optional("_name_which_does_not_exist_")
    with pytest.raises(exception.ModuleNotInstalled):
        module.attribute
