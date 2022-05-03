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


@dataclasses.dataclass
class Complex:
    opt: OptionalArgument
    link: WithLink


def test_simple_schema():
    source = Simple("foo_dataset", "bar_dataset")
    schema = Schema()
    schema.add(Simple, foo=source.foo, bar=source.bar)
    reference = {"simple": {"default": Source(source)}}
    assert schema.sources == reference


def test_two_sources():
    first = Simple("foo1", "bar1")
    second = Simple("foo2", "bar2")
    name = "second_source"
    schema = Schema()
    schema.add(Simple, foo=first.foo, bar=first.bar)
    schema.add(Simple, name=name, foo=second.foo, bar=second.bar)
    reference = {"simple": {"default": Source(first), name: Source(second)}}
    assert schema.sources == reference


def test_file_argument():
    source = Simple("foo_dataset", "bar_dataset")
    filename = "other_file"
    schema = Schema()
    schema.add(Simple, file=filename, foo=source.foo, bar=source.bar)
    reference = {"simple": {"default": Source(source, file=filename)}}
    assert schema.sources == reference


def test_required_argument():
    source = Simple("foo_dataset", "bar_dataset")
    version = RawVersion(1, 2, 3)
    schema = Schema()
    schema.add(Simple, foo=source.foo, bar=source.bar, required=version)
    reference = {"simple": {"default": Source(source, required=version)}}
    assert schema.sources == reference


def test_optional_argument():
    only_mandatory = OptionalArgument("mandatory1")
    name = "mandatory"
    both = OptionalArgument("mandatory2", "optional")
    schema = Schema()
    schema.add(OptionalArgument, name=name, mandatory=only_mandatory.mandatory)
    schema.add(OptionalArgument, mandatory=both.mandatory, optional=both.optional)
    reference = {
        "optional_argument": {name: Source(only_mandatory), "default": Source(both)}
    }
    assert schema.sources == reference


def test_links():
    target = Simple("foo_dataset", "bar_dataset")
    pointer = WithLink("baz_dataset", Link("simple", "default"))
    schema = Schema()
    schema.add(Simple, foo=target.foo, bar=target.bar)
    schema.add(WithLink, baz=pointer.baz, simple=pointer.simple)
    reference = {
        "simple": {"default": Source(target)},
        "with_link": {"default": Source(pointer)},
    }


def test_complex():
    simple = Simple("foo_dataset", "bar_dataset")
    filename = "other_file"
    only_mandatory = OptionalArgument("mandatory1")
    name = "mandatory"
    both = OptionalArgument("mandatory2", "optional")
    pointer = WithLink("baz_dataset", Link("simple", "default"))
    version = RawVersion(1, 2, 3)
    first_complex = Complex(
        Link("optional_argument", "default"),
        Link("with_link", "default"),
    )
    second_complex = Complex(
        Link("optional_argument", name),
        Link("with_link", "default"),
    )
    schema = Schema()
    schema.add(Simple, file=filename, foo=simple.foo, bar=simple.bar)
    schema.add(OptionalArgument, name=name, mandatory=only_mandatory.mandatory)
    schema.add(OptionalArgument, mandatory=both.mandatory, optional=both.optional)
    schema.add(WithLink, required=version, baz=pointer.baz, simple=pointer.simple)
    schema.add(Complex, opt=first_complex.opt, link=first_complex.link)
    schema.add(Complex, name=name, opt=second_complex.opt, link=second_complex.link)
    reference = {
        "simple": {"default": Source(simple, file=filename)},
        "optional_argument": {"default": Source(both), name: Source(only_mandatory)},
        "with_link": {"default": Source(pointer, required=version)},
        "complex": {"default": Source(first_complex), name: Source(second_complex)},
    }
    assert schema.sources == reference
