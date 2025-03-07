# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pytest
from util import VERSION, Mapping, OptionalArgument, Simple, WithLength, WithLink

from py4vasp import exception, raw
from py4vasp._raw.schema import Length, Link, Schema, Source


def test_simple_schema():
    source = Simple("foo_dataset", "bar_dataset")
    schema = Schema(VERSION)
    schema.add(Simple, foo=source.foo, bar=source.bar)
    reference = {"simple": {"default": Source(source)}}
    assert remove_version(schema.sources) == reference


def test_two_sources():
    first = Simple("foo1", "bar1")
    second = Simple("foo2", "bar2")
    name = "second_source"
    schema = Schema(VERSION)
    schema.add(Simple, foo=first.foo, bar=first.bar)
    schema.add(Simple, name=name, foo=second.foo, bar=second.bar)
    reference = {"simple": {"default": Source(first), name: Source(second)}}
    assert remove_version(schema.sources) == reference


def test_file_argument():
    source = Simple("foo_dataset", "bar_dataset")
    filename = "other_file"
    schema = Schema(VERSION)
    schema.add(Simple, file=filename, foo=source.foo, bar=source.bar)
    reference = {"simple": {"default": Source(source, file=filename)}}
    assert remove_version(schema.sources) == reference


def test_required_argument():
    source = Simple("foo_dataset", "bar_dataset")
    version = raw.Version(1, 2, 3)
    schema = Schema(VERSION)
    schema.add(Simple, foo=source.foo, bar=source.bar, required=version)
    reference = {"simple": {"default": Source(source, required=version)}}
    assert remove_version(schema.sources) == reference


def test_optional_argument():
    only_mandatory = OptionalArgument("mandatory1")
    name = "mandatory"
    both = OptionalArgument("mandatory2", "optional")
    schema = Schema(VERSION)
    schema.add(OptionalArgument, name=name, mandatory=only_mandatory.mandatory)
    schema.add(OptionalArgument, mandatory=both.mandatory, optional=both.optional)
    reference = {
        "optional_argument": {name: Source(only_mandatory), "default": Source(both)}
    }
    assert remove_version(schema.sources) == reference


def test_links():
    target = Simple("foo_dataset", "bar_dataset")
    pointer = WithLink("baz_dataset", Link("simple", "default"))
    schema = Schema(VERSION)
    schema.add(Simple, foo=target.foo, bar=target.bar)
    schema.add(WithLink, baz=pointer.baz, simple=pointer.simple)
    reference = {
        "simple": {"default": Source(target)},
        "with_link": {"default": Source(pointer)},
    }
    assert remove_version(schema.sources) == reference


def test_length():
    with_length = WithLength(Length("dataset"))
    schema = Schema(VERSION)
    schema.add(WithLength, num_data=with_length.num_data)
    reference = {"with_length": {"default": Source(with_length)}}
    assert remove_version(schema.sources) == reference


def test_alias():
    first = Simple("foo1", "bar1")
    second = Simple("foo2", "bar2")
    schema = Schema(VERSION)
    schema.add(Simple, foo=first.foo, bar=first.bar, alias=["first", "other"])
    schema.add(Simple, name="second", foo=second.foo, bar=second.bar, alias="more")
    reference = {
        "simple": {
            "default": Source(first),
            "first": Source(first, alias_for="default"),
            "other": Source(first, alias_for="default"),
            "second": Source(second),
            "more": Source(second, alias_for="second"),
        },
    }
    assert remove_version(schema.sources) == reference


def test_custom_data_source():
    def make_data(source):
        pass

    schema = Schema(VERSION)
    schema.add(Simple, file="filename", data_factory=make_data)
    data_factory_source = Source(data=None, file="filename", data_factory=make_data)
    reference = {"simple": {"default": data_factory_source}}
    assert remove_version(schema.sources) == reference


def test_mapping():
    mapping = Mapping("valid_indices", "common_data", "variable_data{}")
    schema = Schema(VERSION)
    schema.add(
        Mapping,
        valid_indices=mapping.valid_indices,
        common=mapping.common,
        variable=mapping.variable,
    )
    reference = {"mapping": {"default": Source(mapping)}}
    assert remove_version(schema.sources) == reference


def remove_version(sources):
    version = sources.pop("version")
    assert version == {"default": Source(VERSION)}
    return sources


def test_file_version():
    schema = Schema(VERSION)
    assert schema.version == VERSION


def test_complex(complex_schema):
    schema, reference = complex_schema
    assert schema.sources == reference


def test_complex_str(complex_schema):
    schema, _ = complex_schema
    reference = """\
---  # schema
version:
    major: major_dataset
    minor: minor_dataset
    patch: patch_dataset

simple:
    default:  &simple-default
        file: other_file
        foo: foo_dataset
        bar: bar_dataset
    factory:  &simple-factory
        file: other_file
        data_factory: complex_schema.<locals>.make_data

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

with_length:
    default:  &with_length-default
        num_data: length(dataset)
    alias_name: *with_length-default

mapping:
    default:  &mapping-default
        valid_indices: foo_mapping
        common: common_data
        variable: variable_data{}
    my_list:  &mapping-my_list
        valid_indices: list_mapping
        common: common
        variable: variable_data_{}

complex:
    default:  &complex-default
        opt: *optional_argument-default
        link: *with_link-default
        mapping: *mapping-default
        length: *with_length-default
    mandatory:  &complex-mandatory
        opt: *optional_argument-mandatory
        link: *with_link-default
        mapping: *mapping-my_list
"""
    assert str(schema) == reference


def test_selections(complex_schema):
    schema, reference = complex_schema
    for quantity, selections in reference.items():
        assert schema.selections(quantity) == selections.keys()


def test_missing_quantity():
    schema = Schema(VERSION)
    schema.add(Simple, foo="foo", bar="bar")
    with pytest.raises(exception.FileAccessError):
        schema.selections(quantity="does not exist")


def test_adding_twice_error():
    schema = Schema(VERSION)
    schema.add(Simple, foo="foo1", bar="bar1")
    with pytest.raises(exception._Py4VaspInternalError):
        schema.add(Simple, foo="foo2", bar="bar2")


def test_schema_is_complete(complex_schema):
    schema, _ = complex_schema
    assert not schema.verified
    schema.verify()  # should not raise error
    assert schema.verified


def test_incomplete_schema():
    target = Simple("foo_dataset", "bar_dataset")
    pointer = WithLink("baz_dataset", Link("simple", source="other"))
    schema = Schema(VERSION)
    schema.add(WithLink, baz=pointer.baz, simple=pointer.simple)
    # test missing quantity
    assert not schema.verified
    with pytest.raises(exception._Py4VaspInternalError):
        schema.verify()
    assert not schema.verified
    # test missing source
    schema.add(Simple, foo=target.foo, bar=target.bar)
    assert not schema.verified
    with pytest.raises(exception._Py4VaspInternalError):
        schema.verify()
    assert not schema.verified
