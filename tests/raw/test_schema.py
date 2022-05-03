from py4vasp.raw import RawVersion
from py4vasp.raw._schema import Schema, Source, Link
import dataclasses


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


def test_required_argument():
    schema = Schema()
    source = Simple("foo_dataset", "bar_dataset")
    version = RawVersion(1, 2, 3)
    schema.add(Simple, foo=source.foo, bar=source.bar, required=version)
    reference = {"simple": {"default": Source(source, required=version)}}
    assert schema.sources == reference


def test_optional_argument():
    schema = Schema()
    only_mandatory = OptionalArgument("mandatory1")
    name = "mandatory"
    both = OptionalArgument("mandatory2", "optional")
    schema.add(OptionalArgument, name=name, mandatory=only_mandatory.mandatory)
    schema.add(OptionalArgument, mandatory=both.mandatory, optional=both.optional)
    reference = {
        "optional_argument": {name: Source(only_mandatory), "default": Source(both)}
    }
    assert schema.sources == reference


def test_links():
    schema = Schema()
    target = Simple("foo_dataset", "bar_dataset")
    pointer = WithLink("baz_dataset", Link("simple", "default"))
    schema.add(Simple, foo=target.foo, bar=target.bar)
    schema.add(WithLink, baz=pointer.baz, simple=pointer.simple)
    reference = {
        "simple": {"default": Source(target)},
        "with_link": {"default": Source(pointer)},
    }
