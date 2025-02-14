# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pytest
from util import (
    VERSION,
    Complex,
    OptionalArgument,
    Sequence,
    Simple,
    WithLength,
    WithLink,
)

from py4vasp import raw
from py4vasp._raw.schema import Length, Link, Schema, Source


@pytest.fixture
def complex_schema():
    def make_data(path):
        return Simple("custom_factory", path)

    simple = Simple("foo_dataset", "bar_dataset")
    filename = "other_file"
    only_mandatory = OptionalArgument("mandatory1")
    name = "mandatory"
    both = OptionalArgument("mandatory2", "optional")
    pointer = WithLink("baz_dataset", Link("simple", "default"))
    version = raw.Version(1, 2, 3)
    length = WithLength(Length("dataset"))
    sequence = Sequence(
        valid_indices="foo_sequence", common="common_data", variable="variable_data{}"
    )
    first = Complex(
        Link("optional_argument", "default"),
        Link("with_link", "default"),
        Link("sequence", "default"),
        Link("with_length", "default"),
    )
    second = Complex(
        Link("optional_argument", name),
        Link("with_link", "default"),
        Link("sequence", "default"),
    )
    schema = Schema(VERSION)
    schema.add(Simple, file=filename, foo=simple.foo, bar=simple.bar)
    schema.add(OptionalArgument, name=name, mandatory=only_mandatory.mandatory)
    schema.add(OptionalArgument, mandatory=both.mandatory, optional=both.optional)
    schema.add(WithLink, required=version, baz=pointer.baz, simple=pointer.simple)
    schema.add(WithLength, alias="alias_name", num_data=length.num_data)
    schema.add(Simple, name="factory", file=filename, data_factory=make_data)
    other_file_source = Source(simple, file=filename)
    data_factory_source = Source(None, file=filename, data_factory=make_data)
    schema.add(
        Sequence,
        valid_indices=sequence.valid_indices,
        common=sequence.common,
        variable=sequence.variable,
    )
    schema.add(
        Complex,
        opt=first.opt,
        link=first.link,
        sequence=first.sequence,
        length=first.length,
    )
    schema.add(
        Complex, name=name, opt=second.opt, link=second.link, sequence=second.sequence
    )
    alias_source = Source(length, alias_for="default")
    reference = {
        "version": {"default": Source(VERSION)},
        "simple": {"default": other_file_source, "factory": data_factory_source},
        "optional_argument": {"default": Source(both), name: Source(only_mandatory)},
        "with_link": {"default": Source(pointer, required=version)},
        "with_length": {"default": Source(length), "alias_name": alias_source},
        "sequence": {"default": Source(sequence)},
        "complex": {"default": Source(first), name: Source(second)},
    }
    return schema, reference
