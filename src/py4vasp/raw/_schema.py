# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses
from typing import Any
from py4vasp import exceptions as exception
from py4vasp import raw
import py4vasp._util.convert as convert


class Schema:
    def __init__(self, version):
        self._sources = {"version": {"default": Source(version)}}
        self._version = version

    def add(self, cls, name="default", file=None, required=None, **kwargs):
        class_name = convert.to_snakecase(cls.__name__)
        self._sources.setdefault(class_name, {})
        if name in self._sources[class_name]:
            message = f"{class_name}/{name} already in the schema. Please choose a different name."
            raise exception.IncorrectUsage(message)
        self._sources[class_name][name] = Source(cls(**kwargs), file, required)

    @property
    def sources(self):
        return self._sources

    @property
    def version(self):
        return self._version

    def __str__(self):
        version = _parse_version(self.version)
        quantities = _parse_quantities(self._sources)
        quantities = "\n".join(quantities)
        return f"""---  # schema
{version}
{quantities}"""


@dataclasses.dataclass
class Source:
    data: Any
    file: str = None
    required: raw.Version = None


@dataclasses.dataclass
class Link:
    quantity: str
    source: str
    __str__ = lambda self: f"*{self.quantity}-{self.source}"


@dataclasses.dataclass
class Length:
    dataset: str
    __str__ = lambda self: f"length({self.dataset})"


def _parse_version(version):
    return f"""version:
    major: {version.major}
    minor: {version.minor}
    patch: {version.patch}"""


def _parse_quantities(quantities):
    for name, sources in quantities.items():
        if name == "version":
            continue
        sources = (_parse_source(name, *source) for source in sources.items())
        yield f"{name}:\n" + "\n".join(sources)


def _parse_source(quantity, source, specification):
    specs = _parse_specification(specification)
    return 4 * " " + f"{source}:  &{quantity}-{source}\n" + "\n".join(specs)


def _parse_specification(specification):
    if specification.file:
        yield 8 * " " + f"file: {specification.file}"
    if specification.required:
        yield 8 * " " + f"required: {_parse_requirement(specification.required)}"
    for field in dataclasses.fields(specification.data):
        key = field.name
        value = getattr(specification.data, key)
        if value:
            yield _parse_field(key, value)


def _parse_field(key, value):
    if isinstance(value, dict):
        value = f"*{value['quantity']}-{value['source']}"
    return 8 * " " + f"{key}: {value}"


def _parse_requirement(version):
    return f"{version.major}.{version.minor}.{version.patch}"
