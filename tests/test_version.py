# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pathlib

import py4vasp


def test_version():
    version = get_version_from_toml_file()
    assert version == py4vasp.__version__


def get_version_from_toml_file():
    root_dir = pathlib.Path(__file__).parent.parent
    with open(root_dir / "pyproject.toml", "r", encoding="utf-8") as toml_file:
        return version_from_lines(toml_file)


def version_from_lines(toml_file):
    for line in toml_file:
        parts = line.split("=")
        if parts[0].strip() == "version":
            return parts[1].strip().strip('"')
