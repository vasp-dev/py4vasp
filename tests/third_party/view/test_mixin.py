# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from unittest.mock import MagicMock

import pytest

from py4vasp._third_party import view

VIEW = MagicMock(spec=view.View)


class ExampleView(view.Mixin):
    def to_view(self):
        VIEW.reset_mock()
        return VIEW


def test_is_abstract_class():
    with pytest.raises(TypeError):
        view.Mixin()


def test_plot_wraps_to_view():
    example = ExampleView()
    assert example.plot() == example.to_view()


def test_converting_view_to_ngl():
    example = ExampleView()
    widget = example.to_ngl()
    VIEW.to_ngl.assert_called_once_with()
    assert widget == VIEW.to_ngl.return_value
