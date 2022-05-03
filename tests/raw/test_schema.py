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


def test_two_sources():
    schema = Schema()
    first = Simple("foo1", "bar1")
    second = Simple("foo2", "bar2")
    name = "second_source"
    schema.add(Simple, foo=first.foo, bar=first.bar)
    schema.add(Simple, name=name, foo=second.foo, bar=second.bar)
    reference = {"simple": {"default": Source(first), name: Source(second)}}
    assert schema.sources == reference


def test_file_argument():
    schema = Schema()
    source = Simple("foo_dataset", "bar_dataset")
    filename = "other_file"
    schema.add(Simple, file=filename, foo=source.foo, bar=source.bar)
    reference = {"simple": {"default": Source(source, file=filename)}}
    assert schema.sources == reference
