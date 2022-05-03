import dataclasses
from typing import Any


class Schema:
    def __init__(self):
        self._sources = {}

    def add(self, cls, name="default", file=None, **kwargs):
        class_name = cls.__name__.lower()
        self._sources.setdefault(class_name, {})
        self._sources[class_name][name] = Source(cls(**kwargs), file)

    @property
    def sources(self):
        return self._sources


@dataclasses.dataclass
class Source:
    source: Any
    file: str = None
