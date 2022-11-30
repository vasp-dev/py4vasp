# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pytest
from util import VERSION, OptionalArgument, Simple, WithLength, WithLink

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
complex:
    default:  &complex-default
        opt: *optional_argument-default
        link: *with_link-default
        length: *with_length-default
    mandatory:  &complex-mandatory
        opt: *optional_argument-mandatory
        link: *with_link-default\
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
    with pytest.raises(exception.IncorrectUsage):
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
    with pytest.raises(AssertionError):
        schema.verify()
    assert not schema.verified
    # test missing source
    schema.add(Simple, foo=target.foo, bar=target.bar)
    assert not schema.verified
    with pytest.raises(AssertionError):
        schema.verify()
    assert not schema.verified
