# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses

import pytest
from util import (
    VERSION,
    Complex,
    ComplexNested,
    Mapping,
    OptionalArgument,
    Simple,
    WithLength,
    WithLink,
    WithOptionalLink,
)

from py4vasp import raw
from py4vasp._raw.schema import DEFAULT_SELECTION, Length, Link, Schema, Source


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
    with_optional = WithOptionalLink(Link("simple", "require_version"))
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
    third = ComplexNested(
        Link("complex", name),
        Link("with_link", "not_so_simple"),
    )
    schema = Schema(VERSION)
    schema.add(Simple, file=filename, **as_dict(simple))
    schema.add(Simple, name="factory", file=filename, data_factory=make_data)
    schema.add(Simple, name="require_version", **as_dict(simple), required=version)
    schema.add(OptionalArgument, name=name, **as_dict(only_mandatory))
    schema.add(OptionalArgument, **as_dict(both))
    schema.add(WithLink, required=version, **as_dict(pointer))
    schema.add(WithLink, name="not_so_simple", required=version, **as_dict(pointer))
    schema.add(WithOptionalLink, **as_dict(with_optional))
    schema.add(WithLength, alias="alias_name", **as_dict(length))
    schema.add(Mapping, **as_dict(mapping))
    schema.add(Mapping, name="my_list", **as_dict(list_))
    schema.add(Complex, **as_dict(first))
    schema.add(Complex, name=name, **as_dict(second))
    schema.add(ComplexNested, name="nested", **as_dict(third))
    other_file_source = Source(simple, file=filename)
    data_factory_source = Source(
        None, file=filename, data_factory=make_data, labels=["factory"]
    )
    alias_source = Source(
        length, alias_for=DEFAULT_SELECTION, labels=[DEFAULT_SELECTION, "alias_name"]
    )
    reference = {
        "version": {DEFAULT_SELECTION: Source(VERSION)},
        "simple": {
            DEFAULT_SELECTION: other_file_source,
            "factory": data_factory_source,
            "require_version": Source(
                simple, required=version, labels=["require_version"]
            ),
        },
        "optional_argument": {
            DEFAULT_SELECTION: Source(both),
            name: Source(only_mandatory, labels=[name]),
        },
        "with_link": {
            DEFAULT_SELECTION: Source(pointer, required=version),
            "not_so_simple": Source(
                pointer, required=version, labels=["not_so_simple"]
            ),
        },
        "with_optional_link": {
            DEFAULT_SELECTION: Source(with_optional),
        },
        "with_length": {
            DEFAULT_SELECTION: Source(length, labels=[DEFAULT_SELECTION, "alias_name"]),
            "alias_name": alias_source,
        },
        "mapping": {
            DEFAULT_SELECTION: Source(mapping, labels=[DEFAULT_SELECTION]),
            "my_list": Source(list_, labels=["my_list"]),
        },
        "complex": {
            DEFAULT_SELECTION: Source(first, labels=[DEFAULT_SELECTION]),
            name: Source(second, labels=[name]),
        },
        "complex_nested": {"nested": Source(third, labels=["nested"])},
    }
    return schema, reference


def as_dict(dataclass):
    # shallow copy of dataclass to dictionary
    return {
        field.name: getattr(dataclass, field.name)
        for field in dataclasses.fields(dataclass)
        if getattr(dataclass, field.name) is not None
    }
