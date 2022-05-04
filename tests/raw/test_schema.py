from py4vasp.raw import RawVersion
from py4vasp.raw._schema import Schema, Source, Link
from util import Simple, OptionalArgument, WithLink, Complex


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


def test_complex(complex_schema):
    schema, reference = complex_schema
    assert schema.sources == reference


def test_complex_str(complex_schema):
    schema, _ = complex_schema
    reference = """\
---  # schema
simple:
    default:  &simple-default
        file: other_file
        foo: foo_dataset
        bar: bar_dataset
optional_argument:
    mandatory:  &optional_argument-mandatory
        mandatory: mandatory1
    default:  &optional_argument-default
        mandatory: mandatory2
        optional: optional
with_link:
    default:  &with_link-default
        required: 1.2.3
        baz: baz_dataset
        simple: *simple-default
complex:
    default:  &complex-default
        opt: *optional_argument-default
        link: *with_link-default
    mandatory:  &complex-mandatory
        opt: *optional_argument-mandatory
        link: *with_link-default\
"""
    assert str(schema) == reference
