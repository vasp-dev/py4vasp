# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""The version of every distribution in this workspace must be identical.

`py4vasp` pins `py4vasp-core` exactly and `vasp-backend` wraps py4vasp internals, so the
three distributions are only ever released together. uv cannot catch a drift for us: the
`[tool.uv.sources]` redirect strips the `==` specifier from the local resolution, so the
pin is only checked once the packages are published. These tests are that check.
"""

import pathlib
import re
import tomllib

import pytest

import py4vasp

ROOT_DIR = pathlib.Path(__file__).parent.parent
META_TOML = ROOT_DIR / "packages" / "py4vasp" / "pyproject.toml"
BACKEND_INIT = ROOT_DIR / "packages" / "backend" / "src" / "vasp" / "backend"


def read_toml(path):
    with open(path, "rb") as file:
        return tomllib.load(file)


def version_of_module(path):
    match = re.search(r'^__version__ = "(.*)"$', path.read_text(), re.MULTILINE)
    assert match is not None, f"no __version__ in {path}"
    return match.group(1)


@pytest.fixture
def expected_version():
    return py4vasp.__version__


def test_core_version_is_dynamic():
    """py4vasp-core takes its version from the module, so there is nothing to sync."""
    core = read_toml(ROOT_DIR / "pyproject.toml")
    assert core["project"]["name"] == "py4vasp-core"
    assert "version" in core["project"]["dynamic"]
    assert core["tool"]["hatch"]["version"]["path"] == "src/py4vasp/__init__.py"


def test_metapackage_version(expected_version):
    assert read_toml(META_TOML)["project"]["version"] == expected_version


def test_metapackage_pins_core_exactly(expected_version):
    dependencies = read_toml(META_TOML)["project"]["dependencies"]
    assert dependencies == [f"py4vasp-core[all]=={expected_version}"]
    assert _requires_core(dependencies[0])


def test_metapackage_extras_pin_core_exactly(expected_version):
    extras = read_toml(META_TOML)["project"].get("optional-dependencies", {})
    checked = []
    for extra, dependencies in extras.items():
        for dependency in dependencies:
            if _requires_core(dependency):
                assert dependency.endswith(f"=={expected_version}"), extra
                checked.append(extra)
    assert checked, "no extra of the wrapper depends on py4vasp-core"


def _requires_core(dependency):
    """Distribution names normalize, so `py4vasp_core` names the same package."""
    name = re.split(r"[\[=<>!~ ;]", dependency, maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", name).lower() == "py4vasp-core"


def test_backend_version(expected_version):
    assert version_of_module(BACKEND_INIT / "__init__.py") == expected_version


def test_backend_pins_core_exactly(expected_version):
    backend = read_toml(ROOT_DIR / "packages" / "backend" / "pyproject.toml")
    assert backend["project"]["dependencies"] == [f"py4vasp-core=={expected_version}"]


def test_metapackage_python_requirement_matches_core():
    """A stricter requires-python in the wrapper would hide py4vasp-core from pip."""
    core = read_toml(ROOT_DIR / "pyproject.toml")["project"]["requires-python"]
    meta = read_toml(META_TOML)["project"]["requires-python"]
    assert meta == core
