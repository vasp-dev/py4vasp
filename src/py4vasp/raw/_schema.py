import dataclasses
from typing import Any
from py4vasp.raw import RawVersion
import py4vasp._util.convert as convert


class Schema:
    def __init__(self):
        self._sources = {}

    def add(self, cls, name="default", file=None, required=None, **kwargs):
        class_name = convert.to_snakecase(cls.__name__)
        self._sources.setdefault(class_name, {})
        self._sources[class_name][name] = Source(cls(**kwargs), file, required)

    @property
    def sources(self):
        return self._sources


@dataclasses.dataclass
class Source:
    source: Any
    file: str = None
    required: RawVersion = None
