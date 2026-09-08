# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Every quantity must offer the uniform surface base.Refinery used to provide.

The Refinery base class gave print, __str__, _repr_pretty_ and selections to all
quantities at once. Composition replaced it, and the methods were silently lost
because nothing checked that a dispatcher implements them. These tests are that
check; they fail for a new quantity that forgets one of them.
"""

import inspect

import pytest

from py4vasp import _calculation, demo, exception
from py4vasp._calculation.dispatch import _REGISTRY


def _registered_quantities():
    for name, entry in sorted(_REGISTRY.items()):
        if isinstance(entry, dict):
            for member, cls in sorted(entry.items()):
                yield f"{name}.{member}", cls
        else:
            yield name, entry


QUANTITIES = list(_registered_quantities())
CLASSES = [cls for _, cls in QUANTITIES]
IDS = [name for name, _ in QUANTITIES]


@pytest.mark.parametrize("cls", CLASSES, ids=IDS)
@pytest.mark.parametrize("method", ("print", "__str__", "_repr_pretty_", "selections"))
def test_every_quantity_defines(cls, method):
    assert method in cls.__dict__


@pytest.mark.parametrize("cls", CLASSES, ids=IDS)
def test_str_takes_an_optional_selection(cls):
    # print forwards its selection to __str__, so every quantity must accept one
    parameter = inspect.signature(cls.__str__).parameters.get("selection")
    assert parameter is not None, f"{cls.__name__}.__str__ takes no selection"
    assert parameter.default is None


def _public_quantities():
    names = list(_calculation.QUANTITIES)
    for group, members in _calculation.GROUPS.items():
        names += [f"{group}.{member}" for member in members]
    return sorted(names)


def _resolve(calculation, name):
    for part in name.split("."):
        calculation = getattr(calculation, part)
    return calculation


@pytest.fixture(scope="module")
def demo_calculation(tmp_path_factory):
    return demo.calculation(tmp_path_factory.mktemp("demo") / "calculation")


@pytest.mark.parametrize("name", _public_quantities())
def test_print_writes_the_string_representation(name, demo_calculation, capsys):
    quantity = _resolve(demo_calculation, name)
    try:
        expected = str(quantity)
    except (exception.NoData, exception.FileAccessError):
        pytest.skip(f"the demo calculation contains no data for {name}")
    assert quantity.print() is None
    assert capsys.readouterr().out == expected + "\n"


@pytest.mark.parametrize("name", _public_quantities())
def test_selections_reports_what_can_be_selected(name, demo_calculation):
    quantity = _resolve(demo_calculation, name)
    try:
        selections = quantity.selections()
    except (exception.NoData, exception.FileAccessError):
        pytest.skip(f"the demo calculation contains no data for {name}")
    # NeighborList deliberately reports a flat list of atom-type pairs; every other
    # quantity maps its own name (and any further keys) to a list of choices
    assert selections
    if isinstance(selections, dict):
        assert all(isinstance(choices, list) for choices in selections.values())
