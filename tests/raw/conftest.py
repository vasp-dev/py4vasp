import pytest
from util import Simple, OptionalArgument, WithLink, Complex
from py4vasp.raw._schema import Schema, Source, Link
from py4vasp.raw import RawVersion


@pytest.fixture
def complex_schema():
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
    return schema, reference
