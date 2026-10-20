# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import importlib.util
import pathlib

import pytest

SCRIPT = pathlib.Path(__file__).parent.parent / "scripts" / "set_version.py"


@pytest.fixture
def set_version(tmp_path):
    spec = importlib.util.spec_from_file_location("set_version", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path
    module.MODULES = (tmp_path / "__init__.py",)
    module.PYPROJECTS = (tmp_path / "pyproject.toml",)
    return module


def write_workspace(root, version):
    (root / "__init__.py").write_text(f'__version__ = "{version}"\n')
    (root / "pyproject.toml").write_text(
        f'version = "{version}"\n'
        f'dependencies = ["py4vasp-core[all]=={version}"]\n'
        f'sphinx = ["py4vasp-core[sphinx]=={version}"]\n'
    )


@pytest.mark.parametrize(
    "old, new",
    [
        ("0.11.3", "0.12.0"),
        ("0.11.3", "1.0.0b1"),
        ("1.0.0b1", "1.0.0rc2"),
        ("1.0.0rc2", "1.0.0"),
    ],
)
def test_set_version(set_version, tmp_path, old, new):
    write_workspace(tmp_path, old)
    set_version.main(new)
    assert (tmp_path / "__init__.py").read_text() == f'__version__ = "{new}"\n'
    assert (tmp_path / "pyproject.toml").read_text() == (
        f'version = "{new}"\n'
        f'dependencies = ["py4vasp-core[all]=={new}"]\n'
        f'sphinx = ["py4vasp-core[sphinx]=={new}"]\n'
    )


@pytest.mark.parametrize("version", ["1.0", "1.0.0-beta", "1.0.0.b1", "v1.0.0"])
def test_reject_invalid_version(set_version, tmp_path, version):
    write_workspace(tmp_path, "0.11.3")
    with pytest.raises(SystemExit):
        set_version.main(version)
