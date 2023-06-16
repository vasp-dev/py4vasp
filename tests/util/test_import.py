# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pytest

from py4vasp import exception
from py4vasp._util import import_


def test_import_not_available():
    module = import_.optional("_name_which_does_not_exist_")
    assert not import_.is_imported(module)
    with pytest.raises(exception.ModuleNotInstalled):
        module.attribute


def test_import_for_existing_module():
    module = import_.optional("py4vasp")
    assert import_.is_imported(module)
