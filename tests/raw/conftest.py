import pytest
from util import Simple, OptionalArgument, WithLink, WithLength, Complex, VERSION
from py4vasp.raw._schema import Schema, Source, Link, Length
from py4vasp import raw


@pytest.fixture
def complex_schema():
    simple = Simple("foo_dataset", "bar_dataset")
    filename = "other_file"
    only_mandatory = OptionalArgument("mandatory1")
    name = "mandatory"
    both = OptionalArgument("mandatory2", "optional")
    pointer = WithLink("baz_dataset", Link("simple", "default"))
    version = raw.Version(1, 2, 3)
    length = WithLength(Length("dataset"))
    first = Complex(
        Link("optional_argument", "default"),
        Link("with_link", "default"),
        Link("with_length", "default"),
    )
    second = Complex(
        Link("optional_argument", name),
        Link("with_link", "default"),
    )
    schema = Schema(VERSION)
    schema.add(Simple, file=filename, foo=simple.foo, bar=simple.bar)
    schema.add(OptionalArgument, name=name, mandatory=only_mandatory.mandatory)
    schema.add(OptionalArgument, mandatory=both.mandatory, optional=both.optional)
    schema.add(WithLink, required=version, baz=pointer.baz, simple=pointer.simple)
    schema.add(WithLength, num_data=length.num_data)
    schema.add(Complex, opt=first.opt, link=first.link, length=first.length)
    schema.add(Complex, name=name, opt=second.opt, link=second.link)
    reference = {
        "version": {"default": Source(VERSION)},
        "simple": {"default": Source(simple, file=filename)},
        "optional_argument": {"default": Source(both), name: Source(only_mandatory)},
        "with_link": {"default": Source(pointer, required=version)},
        "with_length": {"default": Source(length)},
        "complex": {"default": Source(first), name: Source(second)},
    }
    return schema, reference
