# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from __future__ import annotations

import dataclasses
import textwrap

from py4vasp import exception
from py4vasp._util import convert


class Schema:
    def __init__(self, version):
        self._sources = {"version": {"default": Source(version)}}
        self._version = version
        self._verified = False

    def add(self, cls, name="default", file=None, required=None, **kwargs):
        """Add a new quantity to the schema.

        The name of the quantity is deduced from the class you pass in, for example
        DielectricFunction -> dielectric_function. You need to provide where all fields
        of the class can be obtained from in the HDF5 file as keyword arguments. If a
        quantity just links to other objects, use the Link class to specify which
        particular other quantity is used. Furthermore, there are optional arguments
        to set nondefault locations and requirements.

        Parameters
        ----------
        cls : dataclass
            The dataclass defining which data should be read from the HDF5 file.
        name : str
            The name this quantity can be accessed under. This is important if multiple
            entries have the same structure but originate from different data in the
            HDF5 file. Choose the name well, because this is exposed to the user.
        file : str
            If you pass a filename, you overwrite the default behavior from where the
            data should be read. You can use this if VASP produces multiple HDF5 files
            and you want to read from more than one of them.
        required : raw.Version
            Set a version requirement of the HDF5 file that must be fulfilled so that
            the data can be read without error.
        kwargs
            You need to specify for all the fields of the class from where in the HDF5
            file they can be obtained.
        """
        class_name = convert.to_snakecase(cls.__name__)
        self._sources.setdefault(class_name, {})
        if name in self._sources[class_name]:
            message = f"{class_name}/{name} already in the schema. Please choose a different name."
            raise exception.IncorrectUsage(message)
        self._sources[class_name][name] = Source(cls(**kwargs), file, required)
        self._verified = False

    @property
    def sources(self):
        return self._sources

    @property
    def version(self):
        return self._version

    def selections(self, quantity):
        try:
            return self._sources[quantity].keys()
        except KeyError as error:
            raise exception.FileAccessError(error_message(self, quantity)) from error

    @property
    def verified(self):
        return self._verified

    def verify(self):
        "Verify that the schema is complete, i.e., all the links are valid."
        for quantity, sources in self._sources.items():
            for name, source in sources.items():
                self._verify_source(f"{quantity}/{name}", source.data)
        self._verified = True

    def _verify_source(self, key, source):
        for field in dataclasses.fields(source):
            field = getattr(source, field.name)
            if not isinstance(field, Link):
                continue
            self._verify_quantity_is_in_schema(key, field)
            self._verify_source_is_defined_for_quantity(key, field)

    def _verify_quantity_is_in_schema(self, key, field):
        message = f"""Verifying the schema failed in link resolution for {key}, because
    {field.quantity} is not defined in the schema."""
        assert field.quantity in self._sources, message

    def _verify_source_is_defined_for_quantity(self, key, field):
        message = f"""Verifying the schema failed in link resolution for {key}, because
    {field.source} is not a source defined in the schema for the quantity {field.quantity}."""
        assert field.source in self._sources[field.quantity], message

    def __str__(self):
        version = _parse_version(self.version)
        quantities = _parse_quantities(self._sources)
        quantities = "\n".join(quantities)
        return f"""---  # schema
{version}
{quantities}"""


@dataclasses.dataclass
class Source:
    data: Any
    file: str = None
    required: Version = None


@dataclasses.dataclass
class Link:
    quantity: str
    source: str
    __str__ = lambda self: f"*{self.quantity}-{self.source}"


@dataclasses.dataclass
class Length:
    dataset: str
    __str__ = lambda self: f"length({self.dataset})"


def _parse_version(version):
    return f"""version:
    major: {version.major}
    minor: {version.minor}
    patch: {version.patch}"""


def _parse_quantities(quantities):
    for name, sources in quantities.items():
        if name == "version":
            continue
        sources = (_parse_source(name, *source) for source in sources.items())
        yield f"{name}:\n" + "\n".join(sources)


def _parse_source(quantity, source, specification):
    specs = _parse_specification(specification)
    return 4 * " " + f"{source}:  &{quantity}-{source}\n" + "\n".join(specs)


def _parse_specification(specification):
    if specification.file:
        yield 8 * " " + f"file: {specification.file}"
    if specification.required:
        yield 8 * " " + f"required: {_parse_requirement(specification.required)}"
    for field in dataclasses.fields(specification.data):
        key = field.name
        value = getattr(specification.data, key)
        if value:
            yield _parse_field(key, value)


def _parse_field(key, value):
    if isinstance(value, dict):
        value = f"*{value['quantity']}-{value['source']}"
    return 8 * " " + f"{key}: {value}"


def _parse_requirement(version):
    return f"{version.major}.{version.minor}.{version.patch}"


def error_message(schema, quantity, source=None):
    if quantity in schema.sources:
        sources = schema.sources[quantity]
        first_part = f"""\
            py4vasp did not understand your input! The code executed requires to access
            the source="{source}" of the quantity "{quantity}". Perhaps there is a
            spelling mistake in the source? Please, compare the spelling of the source
            "{source}" with the sources py4vasp knows about "{'", "'.join(sources)}"."""
    else:
        first_part = f"""\
            py4vasp does not know how to access the quantity "{quantity}". Perhaps there
            is a spelling mistake? Please, compare the spelling of the quantity "{quantity}"
            with the quantities py4vasp knows about "{'", "'.join(schema.sources)}"."""
    second_part = """\
        It is also possible that this feature was only added in a later version of
        py4vasp, so please check that you use the most recent version."""
    message = textwrap.dedent(first_part) + " " + textwrap.dedent(second_part)
    return "\n" + "\n".join(textwrap.wrap(message, width=80))
