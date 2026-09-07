# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""The `py4vasp` distribution must stay a pure wrapper.

pip extras can only add dependencies, so the code lives in `py4vasp-core` and this
distribution only collects it together with every optional dependency. If it ever started
shipping modules of its own, it would fight with py4vasp-core over site-packages/py4vasp
-- exactly the problem the split removed.
"""

import importlib.metadata

import pytest

import py4vasp

DISTRIBUTION = "py4vasp"


@pytest.fixture
def files():
    files = importlib.metadata.files(DISTRIBUTION)
    if files is None:
        pytest.skip("the distribution does not record its files")
    return files


def test_import_package_is_provided_by_core():
    """A core-only environment still imports py4vasp; this one has both installed."""
    assert py4vasp.__file__ is not None
    assert importlib.metadata.version("py4vasp-core") == py4vasp.__version__


def test_version_agrees_with_the_code():
    assert importlib.metadata.version(DISTRIBUTION) == py4vasp.__version__


def test_distribution_ships_no_modules(files):
    modules = [str(file) for file in files if file.suffix in (".py", ".so", ".pyd")]
    assert modules == []


def test_requires_core_with_all_extras():
    requirements = importlib.metadata.requires(DISTRIBUTION)
    assert any(
        requirement.startswith("py4vasp-core[all]") for requirement in requirements
    )


def test_declared_entry_points():
    names = {entry_point.name for entry_point in _entry_points()}
    assert names == {"py4vasp", "error-analysis", "hugo"}


def test_entry_points_resolve_into_the_other_distribution():
    """Naming another distribution's modules is legal, but nothing checks it for us."""
    for entry_point in _entry_points():
        assert entry_point.value.startswith("py4vasp."), entry_point.value
        assert callable(entry_point.load()), entry_point.name


def _entry_points():
    return importlib.metadata.distribution(DISTRIBUTION).entry_points
