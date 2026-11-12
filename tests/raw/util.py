# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses
from typing import Optional

import h5py

from py4vasp import raw
from py4vasp._raw import mapping
from py4vasp._raw.data import NONE

VERSION = raw.Version("major_dataset", "minor_dataset", "patch_dataset")


@dataclasses.dataclass
class Simple:
    foo: str
    bar: str


@dataclasses.dataclass
class OptionalArgument:
    mandatory: str
    optional: str = NONE()


@dataclasses.dataclass
class WithLink:
    baz: str
    simple: Simple


@dataclasses.dataclass
class WithOptionalLink:
    simple: Optional[Simple] = None


@dataclasses.dataclass
class WithLength:
    num_data: int


@dataclasses.dataclass
class Mapping(mapping.Mapping):
    common: str
    variable: str


@dataclasses.dataclass
class Complex:
    opt: OptionalArgument
    link: WithLink
    mapping: Mapping
    length: WithLength = None


@dataclasses.dataclass
class ComplexNested:
    complex: Complex
    link: WithLink


def write_setup_file(filename, setup, version=(6, 7, 0)):
    """Write an h5 file with the given /setup content.

    `setup` maps a path below /setup to a value or to (value, attributes); a path
    starting with "/" is used as is, e.g. "/metadata/build/variant". Pass
    version=None to omit the version datasets."""
    with h5py.File(filename, "w") as h5f:
        if version is not None:
            for name, number in zip(("major", "minor", "patch"), version):
                h5f[f"version/{name}"] = number
        for key, value in setup.items():
            value, attributes = value if isinstance(value, tuple) else (value, {})
            path = key if key.startswith("/") else f"setup/{key}"
            dataset = h5f.create_dataset(path, data=value)
            dataset.attrs.update(attributes)
    return filename
