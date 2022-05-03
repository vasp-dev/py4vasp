import dataclasses
from typing import Any


class Schema:
    def __init__(self):
        self._sources = {}

    def add(self, cls, name="default", **kwargs):
        class_name = cls.__name__.lower()
        self._sources.setdefault(class_name, {})
        self._sources[class_name][name] = Source(cls(**kwargs))

    @property
    def sources(self):
        return self._sources


@dataclasses.dataclass
class Source:
    source: Any
