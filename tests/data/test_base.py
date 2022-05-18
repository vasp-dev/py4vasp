from py4vasp.data import _base
import dataclasses


@dataclasses.dataclass
class RawData:
    content: str


class Example(_base.Refinery):
    def __post_init__(self):
        self.post_init_called = True

    def read(self):
        return self._raw_data.content


def test_from_raw_data():
    raw_data = RawData("test")
    example = Example.from_data(raw_data)
    assert example.post_init_called
    # access twice too make sure context is regenerated
    assert example.read() == raw_data.content
    assert example.read() == raw_data.content
