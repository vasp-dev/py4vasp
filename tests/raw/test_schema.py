from py4vasp.raw._schema import Schema, Source
import dataclasses


@dataclasses.dataclass
class Simple:
    foo: str
    bar: str


def test_simple_schema():
    schema = Schema()
    source = Simple("foo_dataset", "bar_dataset")
    schema.add(Simple, foo=source.foo, bar=source.bar)
    reference = {"simple": {"default": Source(source)}}
    assert schema.sources == reference
