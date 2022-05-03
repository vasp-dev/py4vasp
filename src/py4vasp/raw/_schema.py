import dataclasses
from typing import Any
from py4vasp.raw import RawVersion


class Schema:
    def __init__(self):
        self._sources = {}

    def add(self, cls, name="default", file=None, required=None, **kwargs):
        class_name = cls.__name__.lower()
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
