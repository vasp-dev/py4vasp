# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses

import pytest
from util import (
    VERSION,
    Complex,
    Mapping,
    OptionalArgument,
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
    mapping = Mapping(
        valid_indices="foo_mapping", common="common_data", variable="variable_data{}"
    )
    list_ = Mapping(
        valid_indices="list_mapping", common="common", variable="variable_data_{}"
    )
    first = Complex(
        Link("optional_argument", "default"),
        Link("with_link", "default"),
        Link("mapping", "default"),
        Link("with_length", "default"),
    )
    second = Complex(
        Link("optional_argument", name),
        Link("with_link", "default"),
        Link("mapping", "my_list"),
    )
    schema = Schema(VERSION)
    schema.add(Simple, file=filename, **as_dict(simple))
    schema.add(Simple, name="factory", file=filename, data_factory=make_data)
    schema.add(OptionalArgument, name=name, **as_dict(only_mandatory))
    schema.add(OptionalArgument, **as_dict(both))
    schema.add(WithLink, required=version, **as_dict(pointer))
    schema.add(WithLength, alias="alias_name", **as_dict(length))
    schema.add(Mapping, **as_dict(mapping))
    schema.add(Mapping, name="my_list", **as_dict(list_))
    schema.add(Complex, **as_dict(first))
    schema.add(Complex, name=name, **as_dict(second))
    other_file_source = Source(simple, file=filename)
    data_factory_source = Source(None, file=filename, data_factory=make_data)
    alias_source = Source(length, alias_for="default")
    reference = {
        "version": {"default": Source(VERSION)},
        "simple": {"default": other_file_source, "factory": data_factory_source},
        "optional_argument": {"default": Source(both), name: Source(only_mandatory)},
        "with_link": {"default": Source(pointer, required=version)},
        "with_length": {"default": Source(length), "alias_name": alias_source},
        "mapping": {"default": Source(mapping), "my_list": Source(list_)},
        "complex": {"default": Source(first), name: Source(second)},
    }
    return schema, reference


def as_dict(dataclass):
    # shallow copy of dataclass to dictionary
    return {
        field.name: getattr(dataclass, field.name)
        for field in dataclasses.fields(dataclass)
        if getattr(dataclass, field.name) is not None
    }
