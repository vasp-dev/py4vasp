# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses

from py4vasp import raw
from py4vasp._raw import mapping

VERSION = raw.Version("major_dataset", "minor_dataset", "patch_dataset")


@dataclasses.dataclass
class Simple:
    foo: str
    bar: str


@dataclasses.dataclass
class OptionalArgument:
    mandatory: str
    optional: str = None


@dataclasses.dataclass
class WithLink:
    baz: str
    simple: Simple


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
