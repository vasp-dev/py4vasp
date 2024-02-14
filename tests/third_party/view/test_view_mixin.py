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


class WithArguments(view.Mixin):
    def to_view(self, mandatory, optional=None):
        VIEW.reset_mock()
        VIEW.arguments = {"mandatory": mandatory, "optional": optional}
        return VIEW


def test_arguments_passed_by_plot():
    example = WithArguments()
    view = example.plot("only mandatory")
    assert view.arguments == {"mandatory": "only mandatory", "optional": None}
    view = example.plot("first", "second")
    assert view.arguments == {"mandatory": "first", "optional": "second"}
    view = example.plot(optional="foo", mandatory="bar")
    assert view.arguments == {"mandatory": "bar", "optional": "foo"}


def test_arguments_passed_by_to_ngl():
    example = WithArguments()
    example.to_ngl("only mandatory")
    assert VIEW.arguments == {"mandatory": "only mandatory", "optional": None}
    example.to_ngl("first", "second")
    assert VIEW.arguments == {"mandatory": "first", "optional": "second"}
    example.to_ngl(optional="foo", mandatory="bar")
    assert VIEW.arguments == {"mandatory": "bar", "optional": "foo"}
