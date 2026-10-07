# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Tests for Calculation wiring to _REGISTRY and _source storage."""

import ast
import contextlib
import pathlib
import subprocess
import sys
import zipfile
from unittest.mock import patch

import pytest

import py4vasp
from py4vasp import Calculation, _calculation
from py4vasp._calculation.dispatch import (
    _REGISTRY,
    DataSource,
    DictSource,
    FileSource,
    Group,
    quantity,
)


@contextlib.contextmanager
def _isolated_registry():
    """Restore _REGISTRY to its original state after the block."""
    saved = {k: dict(v) if isinstance(v, dict) else v for k, v in _REGISTRY.items()}
    try:
        yield
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(saved)


class _FakeDispatcher:
    _quantity_name = "fake_qty"

    def __init__(self, source, quantity_name="fake_qty"):
        self.source = source
        self.quantity_name = quantity_name


class _FakeGroupDispatcher:
    _quantity_name = "fake_member"

    def __init__(self, source, quantity_name="fake_member"):
        self.source = source
        self.quantity_name = quantity_name


class TestCalculationStoresSource:
    def test_from_path_stores_file_source(self, tmp_path):
        calc = Calculation.from_path(tmp_path)
        assert isinstance(calc._source, FileSource)
        assert calc._source.path == tmp_path.resolve()

    def test_from_path_file_source_has_no_file(self, tmp_path):
        calc = Calculation.from_path(tmp_path)
        assert calc._source._file is None

    def test_from_file_stores_file_source(self, tmp_path):
        file = tmp_path / "vaspout.h5"
        calc = Calculation.from_file(file)
        assert isinstance(calc._source, FileSource)

    def test_from_file_source_path_is_parent_dir(self, tmp_path):
        file = tmp_path / "vaspout.h5"
        calc = Calculation.from_file(file)
        assert calc._source.path == tmp_path.resolve()

    def test_from_file_source_file_is_forwarded(self, tmp_path):
        file = tmp_path / "vaspout.h5"
        calc = Calculation.from_file(file)
        # only the name, because the directory is already part of the source path
        assert calc._source._file == "vaspout.h5"

    def test_path_property_still_works_after_from_path(self, tmp_path):
        calc = Calculation.from_path(tmp_path)
        assert calc.path() == tmp_path.resolve()

    def test_path_property_still_works_after_from_file(self, tmp_path):
        file = tmp_path / "vaspout.h5"
        calc = Calculation.from_file(file)
        assert calc.path() == tmp_path.resolve()


class TestCalculationGetattr:
    def test_getattr_top_level_returns_registered_dispatcher(self, tmp_path):
        with _isolated_registry():

            @quantity("fake_qty")
            class FakeDispatcher(_FakeDispatcher):
                pass

            calc = Calculation.from_path(tmp_path)
            result = calc.fake_qty
            assert isinstance(result, FakeDispatcher)

    def test_getattr_dispatcher_receives_source(self, tmp_path):
        with _isolated_registry():

            @quantity("fake_qty2")
            class FakeDispatcher2(_FakeDispatcher):
                _quantity_name = "fake_qty2"

                def __init__(self, source, quantity_name="fake_qty2"):
                    self.source = source
                    self.quantity_name = quantity_name

            calc = Calculation.from_path(tmp_path)
            result = calc.fake_qty2
            assert result.source is calc._source

    def test_getattr_group_returns_group_instance(self, tmp_path):
        with _isolated_registry():

            @quantity("fake_member", group="fake_group")
            class FakeGroupDispatcher(_FakeGroupDispatcher):
                pass

            calc = Calculation.from_path(tmp_path)
            result = calc.fake_group
            assert isinstance(result, Group)

    def test_getattr_group_member_instantiates_dispatcher(self, tmp_path):
        with _isolated_registry():

            @quantity("fake_member2", group="fake_group2")
            class FakeGroupDispatcher2(_FakeGroupDispatcher):
                _quantity_name = "fake_member2"

                def __init__(self, source, quantity_name="fake_member2"):
                    self.source = source
                    self.quantity_name = quantity_name

            calc = Calculation.from_path(tmp_path)
            result = calc.fake_group2.fake_member2
            assert isinstance(result, FakeGroupDispatcher2)

    def test_getattr_unknown_attribute_raises_attribute_error(self, tmp_path):
        calc = Calculation.from_path(tmp_path)
        with pytest.raises(AttributeError):
            calc.this_quantity_does_not_exist_in_any_registry

    def test_getattr_dispatcher_source_path_matches_calculation_path(self, tmp_path):
        with _isolated_registry():

            @quantity("fake_qty3")
            class FakeDispatcher3(_FakeDispatcher):
                _quantity_name = "fake_qty3"

                def __init__(self, source, quantity_name="fake_qty3"):
                    self.source = source
                    self.quantity_name = quantity_name

            calc = Calculation.from_path(tmp_path)
            assert calc.fake_qty3.source.path == calc._path

    def test_new_arch_quantities_accessible_via_getattr(self):
        """New-arch quantities registered via @quantity are accessible via __getattr__."""
        with patch("py4vasp.raw.access"):
            calc = Calculation.from_path(".")
            # 'energy' is a new-arch quantity wired via _REGISTRY/__getattr__
            assert hasattr(calc, "energy")


class TestCalculationDir:
    def test_registry_quantities_appear_in_dir(self, tmp_path):
        with _isolated_registry():

            @quantity("fake_dir_qty")
            class FakeDirDispatcher(_FakeDispatcher):
                _quantity_name = "fake_dir_qty"

                def __init__(self, source, quantity_name="fake_dir_qty"):
                    self.source = source
                    self.quantity_name = quantity_name

            calc = Calculation.from_path(tmp_path)
            assert "fake_dir_qty" in dir(calc)

    def test_registry_groups_appear_in_dir(self, tmp_path):
        with _isolated_registry():

            @quantity("fake_dir_member", group="fake_dir_group")
            class FakeDirGroupDispatcher(_FakeDispatcher):
                _quantity_name = "fake_dir_member"

                def __init__(self, source, quantity_name="fake_dir_member"):
                    self.source = source
                    self.quantity_name = quantity_name

            calc = Calculation.from_path(tmp_path)
            assert "fake_dir_group" in dir(calc)

    def test_existing_attributes_still_in_dir(self, tmp_path):
        calc = Calculation.from_path(tmp_path)
        names = dir(calc)
        assert "from_path" in names
        assert "from_file" in names


_PRINT_REGISTRY_VIEWS = """
import sys
sys.path.insert(0, sys.argv[1])
from py4vasp import _calculation
print(repr((_calculation.QUANTITIES, _calculation.GROUPS)))
"""


def test_quantities_discovered_when_imported_from_zip(tmp_path):
    # Frozen builds (PyInstaller, zipimport) provide the modules without .py files on
    # disk, so discovery must go through the package loader.
    package_dir = pathlib.Path(py4vasp.__file__).parent
    archive = tmp_path / "py4vasp.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        # directories are written too, so namespace packages (no __init__.py) resolve
        for source in [package_dir, *package_dir.rglob("*")]:
            if source.is_dir() or source.suffix == ".py":
                zip_file.write(source, source.relative_to(package_dir.parent))
    output = subprocess.run(
        [sys.executable, "-I", "-c", _PRINT_REGISTRY_VIEWS, str(archive)],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    ).stdout
    quantities, groups = ast.literal_eval(output)
    assert quantities == _calculation.QUANTITIES
    assert groups == _calculation.GROUPS


def test_warns_when_no_quantity_modules_found():
    with patch("py4vasp._calculation.pkgutil.iter_modules", return_value=[]):
        with pytest.warns(UserWarning, match="py4vasp._calculation"):
            _calculation._ensure_all_quantities_imported()
