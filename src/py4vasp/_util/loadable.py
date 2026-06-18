# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Utilities to determine loadable quantity selections for a Calculation."""

import dataclasses
import inspect

import h5py
import numpy as np

from py4vasp import exception, raw
from py4vasp._raw.definition import DEFAULT_FILE, schema, unique_selections
from py4vasp._raw.schema import DEFAULT_SELECTION, Length, Link
from py4vasp._util import check, convert


def loadable_selections(
    calculation,
    call_name,
    schema_name,
    method,
    open_files,
    stack,
    cache,
    legacy_quantities,
):
    """Return loadable selections for one quantity as {source: snippet}."""
    method_name = method or "read"
    try:
        sources = list(unique_selections(schema_name))
    except exception.FileAccessError:
        return {}
    if not _implements(calculation, call_name, method_name):
        return {}
    convention = _source_convention(calculation, call_name, legacy_quantities)
    loadable = {}
    for source_name in sources:
        conv = "plain" if source_name == DEFAULT_SELECTION else convention
        if conv is None:
            # The source cannot be addressed yet (e.g. a not-yet-migrated quantity).
            continue
        if not _confirm_selection(
            calculation,
            call_name,
            schema_name,
            source_name,
            method_name,
            conv,
            open_files,
            stack,
            cache,
            legacy_quantities,
        ):
            continue
        loadable[source_name] = _call_snippet(call_name, method_name, source_name, conv)
    return loadable


def _implements(calculation, call_name, method_name):
    try:
        quantity = _quantity_object(calculation, call_name)
    except Exception:
        return False
    return callable(getattr(quantity, method_name, None))


def _source_convention(calculation, call_name, legacy_quantities):
    """Return how a non-default source is addressed: 'keyword', 'index', or None."""
    try:
        quantity = _quantity_object(calculation, call_name)
    except Exception:
        return None
    read = getattr(quantity, "read", None)
    if read is not None and _accepts_selection(read):
        return "keyword"
    if call_name not in legacy_quantities and hasattr(type(quantity), "__getitem__"):
        return "index"
    return None


def _confirm_selection(
    calculation,
    call_name,
    schema_name,
    source_name,
    method_name,
    convention,
    open_files,
    stack,
    cache,
    legacy_quantities,
):
    if method_name == "read":
        return _confirm_read(
            calculation,
            call_name,
            schema_name,
            source_name,
            open_files,
            stack,
            cache,
            legacy_quantities,
        )
    # For explicit methods, successful invocation proves the source can be loaded.
    if (
        _schema_satisfied(
            calculation,
            schema_name,
            source_name,
            open_files,
            stack,
            cache,
            legacy_quantities,
        )
        is False
    ):
        return False
    return _invoke(
        calculation,
        call_name,
        method_name,
        source_name,
        legacy_quantities,
        convention,
    )


def _confirm_read(
    calculation,
    call_name,
    schema_name,
    source_name,
    open_files,
    stack,
    cache,
    legacy_quantities,
):
    key = (schema_name, source_name)
    if key in cache:
        return cache[key]
    cache[key] = True
    verdict = _schema_satisfied(
        calculation,
        schema_name,
        source_name,
        open_files,
        stack,
        cache,
        legacy_quantities,
    )
    if verdict is None:
        verdict = _invoke(
            calculation,
            call_name,
            "read",
            source_name,
            legacy_quantities,
        )
    cache[key] = bool(verdict)
    return cache[key]


def _schema_satisfied(
    calculation,
    schema_name,
    source_name,
    open_files,
    stack,
    cache,
    legacy_quantities,
):
    try:
        source = schema.sources[schema_name][source_name]
    except KeyError:
        return False
    if source.required is not None:
        version = _file_version(calculation, open_files, stack)
        if version is None or version < source.required:
            return False
    filename = calculation._file or source.file or DEFAULT_FILE
    if source.data is None:
        return (calculation._path / filename).exists()
    h5f = _open_h5(calculation, open_files, stack, filename)
    if h5f is None:
        return False
    return _fields_present(
        calculation, source.data, h5f, open_files, stack, cache, legacy_quantities
    )


def _fields_present(
    calculation,
    data,
    h5f,
    open_files,
    stack,
    cache,
    legacy_quantities,
):
    indices = _valid_indices(data, h5f)
    field_names = {field.name for field in dataclasses.fields(data)}
    if "valid_indices" in field_names and not indices:
        return False
    missing = False
    for field in dataclasses.fields(data):
        if field.name == "valid_indices" or _is_optional(field):
            continue
        value = getattr(data, field.name)
        if check.is_none(value):
            continue
        if isinstance(value, Link):
            if not _confirm_read(
                calculation,
                value.quantity.lstrip("_"),
                value.quantity,
                value.source,
                open_files,
                stack,
                cache,
                legacy_quantities,
            ):
                return False
            continue
        if not _value_present(value, h5f, indices):
            missing = True
    return None if missing else True


def _value_present(value, h5f, indices):
    if isinstance(value, Length):
        return h5f.get(value.dataset) is not None
    path = str(value)
    if "{}" in path:
        if not indices:
            return False
        return h5f.get(_format_index(path, indices[0])) is not None
    return h5f.get(path) is not None


def _valid_indices(data, h5f):
    field_names = {field.name for field in dataclasses.fields(data)}
    if "valid_indices" not in field_names:
        return None
    dataset = h5f.get(str(getattr(data, "valid_indices")))
    if dataset is None:
        return None
    raw_value = dataset[()]
    if np.ndim(raw_value) == 0:
        return list(range(int(raw_value)))
    return [convert.text_to_string(index) for index in raw_value]


def _file_version(calculation, open_files, stack):
    h5f = _open_h5(calculation, open_files, stack, calculation._file or DEFAULT_FILE)
    if h5f is None:
        return None
    try:
        return raw.Version(
            int(h5f[schema.version.major][()]),
            int(h5f[schema.version.minor][()]),
            int(h5f[schema.version.patch][()]),
        )
    except (KeyError, OSError, TypeError, ValueError):
        return None


def _open_h5(calculation, open_files, stack, filename):
    if filename in open_files:
        return open_files[filename]
    try:
        handle = stack.enter_context(h5py.File(calculation._path / filename, "r"))
    except (FileNotFoundError, OSError):
        handle = None
    open_files[filename] = handle
    return handle


def _invoke(
    calculation,
    call_name,
    method_name,
    source_name,
    legacy_quantities,
    convention=None,
):
    try:
        quantity = _quantity_object(calculation, call_name)
    except Exception:
        return False
    method = getattr(quantity, method_name, None)
    if method is None:
        return False
    if source_name == DEFAULT_SELECTION:
        return _call_succeeds(method)
    if convention is None:
        convention = _source_convention(calculation, call_name, legacy_quantities)
    if convention == "keyword":
        return _call_succeeds(lambda: method(selection=source_name))
    if convention == "index":
        return _call_succeeds(lambda: getattr(quantity[source_name], method_name)())
    return False


def _quantity_object(calculation, call_name):
    if "." in call_name:
        group_name, member = call_name.split(".", 1)
        return getattr(getattr(calculation, group_name), member)
    return getattr(calculation, call_name)


def _is_optional(field):
    annotation = field.type
    if not isinstance(annotation, str):
        annotation = getattr(annotation, "__name__", str(annotation))
    annotation = annotation.strip()
    return annotation.startswith("Optional") or annotation.startswith("typing.Optional")


def _format_index(path, index):
    if isinstance(index, (int, np.integer)):
        index = int(index) + 1
    return path.format(index)


def _accepts_selection(func):
    try:
        return "selection" in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def _call_succeeds(func):
    try:
        func()
        return True
    except Exception:
        return False


def _call_snippet(call_name, method_name, source_name, convention):
    access = f"calculation.{call_name}"
    if convention == "keyword":
        return f"{access}.{method_name}(selection={source_name!r})"
    if convention == "index":
        return f"{access}[{source_name!r}].{method_name}()"
    return f"{access}.{method_name}()"
