import dataclasses
from typing import Any
from py4vasp.raw import RawVersion
import py4vasp._util.convert as convert


class Schema:
    def __init__(self, version):
        self._sources = {}
        self.version = version

    def add(self, cls, name="default", file=None, required=None, **kwargs):
        class_name = convert.to_snakecase(cls.__name__)
        self._sources.setdefault(class_name, {})
        self._sources[class_name][name] = Source(cls(**kwargs), file, required)

    @property
    def sources(self):
        return self._sources

    def __str__(self):
        version = _parse_version(self.version)
        quantities = (_parse_quantity(*quantity) for quantity in self._sources.items())
        quantities = "\n".join(quantities)
        return f"""---  # schema
{version}
{quantities}"""


@dataclasses.dataclass
class Source:
    data: Any
    file: str = None
    required: RawVersion = None


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


def _parse_quantity(name, sources):
    sources = (_parse_source(name, *source) for source in sources.items())
    return f"{name}:\n" + "\n".join(sources)


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
