# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pytest

from py4vasp._third_party import view


class ExampleView(view.Mixin):
    pass


def test_is_abstract_class():
    with pytest.raises(TypeError):
        view.Mixin()
