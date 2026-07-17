# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import ast
import functools
import inspect
from contextlib import suppress
from math import gcd
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from h5py import File

from py4vasp import exception
from py4vasp._raw.data import Version
from py4vasp._raw.definition import DEFAULT_SOURCE, Schema, unique_selections
from py4vasp._raw.models import parse_schema_version
from py4vasp._raw.schema import Length, Link
from py4vasp._util import convert


def combine_db_dicts(*args) -> dict:
    """Combine a list of dictionaries safely.

    This function deep-merges multiple dictionaries representing database entries.
    If there are overlapping keys:
    - If the values are both dictionaries, they are merged recursively.
    - If the values are not dictionaries and are equal, nothing is changed.
    - If the values are not dictionaries and differ, the right-most / newest value overwrites the previous.

    Parameters
    ----------
    *args : dict
        The dictionaries to combine.

    Returns
    -------
    dict
        A new dictionary containing the combined entries.
    """
    base_dict = {}
    for arg in args:
        base_dict = _merge_dicts(base_dict, arg)
    return base_dict


def _merge_dicts(dict1: dict, dict2: dict) -> dict:
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        raise TypeError("Both arguments must be dictionaries.")
    for key2, value2 in dict2.items():
        if key2 in dict1:
            value1 = dict1[key2]
            if isinstance(value1, dict) and isinstance(value2, dict):
                dict1[key2] = _merge_dicts(value1, value2)
            elif isinstance(value1, dict) or isinstance(value2, dict):
                raise exception._Py4VaspInternalError(
                    f"Database Dictionary Merge Conflict at key '{key2}': one value is a dictionary, the other is not."
                )
            elif value1 != value2:
                dict1[key2] = value2
        else:
            dict1[key2] = value2
    return dict1


def construct_database_data_key(
    group_name, quantity_name, selection
) -> Tuple[str, bool]:
    "Construct the key for storing database data."
    has_selection = (
        selection is not None and selection != "" and selection != DEFAULT_SOURCE
    )
    full_key = quantity_name.lstrip("_") + (f":{selection}" if has_selection else "")
    if group_name is not None:
        full_key = f"{group_name}.{full_key}"
    return full_key, has_selection


@functools.cache
def should_load(
    quantity: str,
    source: str,
    h5file: File,
    schema: Schema,
    version: Optional[Version] = None,
    print_debug: bool = False,
) -> tuple[bool, Optional[Version], bool, list[str]]:
    should_load_ = True
    should_attempt_read = False
    additional_keys = []
    if print_debug:
        print(f"[CHECK] Checking availability of quantity {quantity}:{source}.")
    if quantity in schema.sources:
        required = schema.sources[quantity][source].required
        check_success, version = check_version(h5file, required, schema, version)
        if not check_success:
            should_load_ = False
            if print_debug:
                print(
                    f"[CHECK-Should-Load] VASP version too old for {quantity}:{source}."
                )
        source_info = schema.sources[quantity][source].data
        if source_info is not None:
            quantity_dict = {}
            for key, _ in source_info.__dataclass_fields__.items():
                link = getattr(source_info, key)
                if isinstance(link, Link):
                    quantity_dict[key], version, _, subquantity_additional_keys = (
                        should_load(
                            link.quantity,
                            link.source,
                            h5file,
                            schema,
                            version,
                            print_debug,
                        )
                    )
                    quantity_dict[key] = (
                        True  # overwrite for Link quantities on should_load
                    )
                    additional_keys.append(f"{link.quantity}:{link.source}")
                    additional_keys.extend(subquantity_additional_keys)
                elif isinstance(link, Length):
                    quantity_dict[key] = link.dataset in h5file
                elif link is not None and isinstance(link, str):
                    quantity_dict[key] = link in h5file
            quantity_dict = {k: v for k, v in quantity_dict.items() if v is True}
            try:
                new_class = source_info.__class__(**quantity_dict)
            except Exception as e:
                if print_debug:
                    print(
                        "[CHECK] Error when instantiating class:",
                        source_info.__class__,
                        quantity_dict,
                        e,
                    )
                should_load_ = False
        else:
            should_load_ = False
            should_attempt_read = True
            # data_factory might be set
            if schema.sources[quantity][source].data_factory is not None:
                _, version, _, additional_keys = should_load(
                    quantity, DEFAULT_SOURCE, h5file, schema, version, print_debug
                )
    if not should_load_ and print_debug:
        print(f"[CHECK] Quantity {quantity} with source {source} is not available.")
    return should_load_, version, should_attempt_read, additional_keys


def check_version(h5f, required, schema, current_version=None):
    if not required:
        return True, current_version
    if current_version is None:
        current_version = Version(
            major=h5f[schema.version.major][()],
            minor=h5f[schema.version.minor][()],
            patch=h5f[schema.version.patch][()],
        )
    return not (current_version < required), current_version


def clean_db_key(
    key: str, db_key_suffix: Optional[str] = None, group_name: Optional[str] = None
) -> str:
    key = (
        (key.lstrip("_") + (db_key_suffix or ""))
        if not (":" in key)
        else key.lstrip("_")
    )
    if group_name is not None and key.startswith(group_name):
        key_first, key_second = key[: len(group_name)], key[len(group_name) :]
        if key_second.startswith("_"):
            key = f"{key_first}.{key_second.lstrip('_')}"
    return key


def clean_db_dict_keys(dict_to_clean: dict, rid_default_selection: bool = True) -> dict:
    if rid_default_selection:
        # Fix keys to remove default selection suffixes
        dict_to_clean = dict(
            zip(
                [
                    (
                        key
                        if not (key.endswith(f":{DEFAULT_SOURCE}"))
                        else key[: -len(f":{DEFAULT_SOURCE}")]
                    )
                    for key in dict_to_clean.keys()
                ],
                dict_to_clean.values(),
            )
        )

    return dict_to_clean


def get_primitive_ion_numbers(
    number_ion_types: list[int],
) -> list[int]:
    """Calculate the number of ions in the primitive cell.

    Parameters
    ----------
    number_ion_types : list[int]
        Number of ions of each type in the conventional cell.

    Returns
    -------
    list[int]
        Number of ions of each type in the primitive cell.
    """
    if number_ion_types is None:
        return None

    _gcd = functools.reduce(gcd, number_ion_types)
    return [n // _gcd for n in number_ion_types]


def get_formula_and_compound(
    ion_types: list[str], number_ion_types: list[int]
) -> Tuple[str, str]:
    """Generate the chemical formula and compound name.

    Parameters
    ----------
    ion_types : list[str]
        List of unique ion types.
    number_ion_types : list[int]
        Number of ions of each type.

    Returns
    -------
    tuple[str, str, list[str], list[int], list[int]]
        The chemical formula and compound name, followed by unique ion types, their
        counts in the conventional cell, and their counts in the primitive cell.

    When the element names are unavailable but the counts are known, the counts (and
    their primitive reduction) are still returned; the name-derived formula, compound,
    and unique-type list are ``None`` because they cannot be built without names.
    """
    if number_ion_types is None:
        return None, None, None, None, None
    if ion_types is None:
        # No element names: keep the counts in their raw order (they cannot be
        # aggregated or sorted by element) and reduce them to the primitive cell.
        simple_numbers = list(number_ion_types)
        return (
            None,
            None,
            None,
            simple_numbers,
            get_primitive_ion_numbers(simple_numbers),
        )

    formula_dict = {}
    # first sort, then count up numbers in case any ion type is non-unique
    sorted_types = sorted(zip(ion_types, number_ion_types), key=lambda x: x[0])
    for ion_type, number in sorted_types:
        if number > 0:
            formula_dict[ion_type] = formula_dict.get(ion_type, 0) + number
    simple_numbers = list(formula_dict.values())

    # now compute primitive numbers and set in formula_dict
    primitive_numbers = get_primitive_ion_numbers(simple_numbers)
    for k, n in zip(formula_dict.keys(), primitive_numbers):
        formula_dict[k] = n
    # compound and formula are constructed from the primitive counts
    compound_parts = list(formula_dict.keys())
    formula_parts = [
        f"{ion}{formula_dict[ion] if formula_dict[ion] > 1 else ''}"
        for ion in compound_parts
    ]
    formula = "".join(formula_parts)
    compound = "-".join(compound_parts)
    return (
        formula,
        compound,
        list(formula_dict.keys()),
        simple_numbers,
        primitive_numbers,
    )


def get_dataclass_fields(dataclass: Any) -> List[dict]:
    """Get the fields of a dataclass as a list of dictionaries.

    Parameters
    ----------
    dataclass : Any
        The dataclass to get the fields from.

    Returns
    -------
    List[dict]
        A list of dictionaries, each containing the name, type, and
        optional field documentation of a dataclass field.
    """
    from dataclasses import fields

    dataclass_fields = fields(dataclass)
    docstrings = _get_dataclass_field_docstrings(dataclass)
    return [
        {
            "name": field.name,
            "type": field.type,
            "documentation": docstrings.get(field.name),
        }
        for field in dataclass_fields
    ]


def _get_dataclass_field_docstrings(dataclass: Any) -> Dict[str, Optional[str]]:
    """Extract per-field documentation from a dataclass using AST.

    Expects field documentation to be provided as a string expression directly
    below the corresponding field definition.
    """
    with suppress(Exception):
        source_file = inspect.getsourcefile(dataclass)
        if source_file is None:
            return {}
        source = Path(source_file).read_text()
        tree = ast.parse(source)
        class_node = _find_class_node(tree, dataclass.__name__)
        if class_node is None:
            return {}

        docstrings: Dict[str, Optional[str]] = {}
        class_body = class_node.body
        for index, node in enumerate(class_body):
            field_name = _extract_field_name(node)
            if field_name is None:
                continue
            docstrings[field_name] = _extract_following_docstring(class_body, index)
        return docstrings

    return {}


def _find_class_node(tree: ast.AST, class_name: str) -> Optional[ast.ClassDef]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


def _extract_field_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        if isinstance(target, ast.Name):
            return target.id
    return None


def _extract_following_docstring(
    class_body: List[ast.stmt], index: int
) -> Optional[str]:
    next_index = index + 1
    if next_index >= len(class_body):
        return None
    next_node = class_body[next_index]
    if not isinstance(next_node, ast.Expr):
        return None
    doc = _get_constant_value(next_node.value)
    return doc if isinstance(doc, str) else None


def get_all_possible_keys(
    to_print: bool = False, debug: bool = False
) -> tuple[Dict[str, List[Tuple[str, str]]], Dict[str, Optional[str]]]:
    """
    Extract all possible keys from _to_database methods in calculation classes.

    Returns
    -------
        Tuple[Dict[str, List[Tuple[str, str]]], Dict[str, Optional[str]]]
        A tuple:
                    - A dictionary where keys are database dataclass names (e.g. BandModel) and
                        values are lists of tuples containing (attribute_name, attribute_type).
                    - A dictionary mapping all available keys (group.quantity[:selection]) to
                        their dataclass names. If no matching dataclass exists, the value is None.
                        The default selection is represented without a suffix, e.g. "band"
                        instead of "band:default".
    """
    all_keys = {}

    from py4vasp._calculation import GROUPS, QUANTITIES

    # Enumerate the public quantities (and group members) directly from the registry.
    # The actual database keys/types are resolved downstream from the schema selections
    # and the *Model dataclasses.
    candidate_keys = list(QUANTITIES)
    candidate_keys += [
        f"{group}_{member}" for group, members in GROUPS.items() for member in members
    ]
    for key in candidate_keys:
        all_keys[key] = []

    output_type_dict: dict[str, Optional[str]] = {}
    """Map all available keys (group.quantity[:selection]) to dataclass names."""

    for k in list(all_keys.keys()):
        selections_list = []
        with suppress(exception.Py4VaspError):
            selections_list = unique_selections(k)
        constructed_key = _quantity_label_to_db_key(k)
        dataclass_name = _get_dataclass_name_for_quantity(k)
        for sel in selections_list:
            if sel in GROUPS.keys():
                output_type_dict[f"{sel}._{k}"] = dataclass_name
        if len(selections_list) == 0:
            # Derived quantities such as `optics` have no schema/raw data of their own.
            # Record them so the mapping stays complete. A derived quantity may still
            # have a database representation (its own *Model dataclass) via a hand-written
            # `_to_database`; keep those so the dataclass is enumerated. Drop only the
            # ones without any database dataclass from the storage keys.
            output_type_dict[constructed_key] = dataclass_name
            if dataclass_name is None:
                all_keys.pop(k)
        else:
            selections_list = [
                sel for sel in selections_list if sel not in GROUPS.keys()
            ]
            non_default_selections = [
                sel for sel in selections_list if sel != DEFAULT_SOURCE
            ]
            output_type_dict[constructed_key] = dataclass_name
            for sel in non_default_selections:
                output_type_dict[f"{constructed_key}:{sel}"] = dataclass_name

    # `energy` is special: its format -- and thus its database model -- is detected from
    # the data at runtime, so the single `energy` quantity is represented by three
    # different models rather than one. The auto-discovery above cannot resolve this, so
    # map the selections explicitly: the default source is a relaxation or an MD run
    # (relaxation is the representative), while the `afqmc` selection always maps to
    # EnergyAfqmcModel. All three models are injected into ``main_keys`` below so their
    # fields still appear in the generated documentation.
    energy_db_models = ("EnergyRelaxationModel", "EnergyMDModel", "EnergyAfqmcModel")
    if "energy" in all_keys:
        output_type_dict["energy"] = "EnergyRelaxationModel"
        output_type_dict["energy:afqmc"] = "EnergyAfqmcModel"

    sort_keys_list = ["energy"]

    if to_print:
        print("\n--- PARSED KEYS: ---")
        for k, v in sorted(all_keys.items()):
            if v is not None and len(v) > 0:
                print(f"\t{k}:")
                should_sort = k in sort_keys_list
                vsort = sorted(v) if should_sort else v
                for subkey in vsort:
                    print(f"\t\t- {subkey}")

        print("\n--- EMPTY KEYS ---")
        for k, v in sorted(all_keys.items()):
            if v is not None and len(v) == 0:
                print(f"\t{k}")

        print("\n--- MISSING _to_database ---")
        for k, v in sorted(all_keys.items()):
            if v is None:
                print(f"\t{k}")

        print("\n--- OUTPUT TYPE DICT ---")
        for k, v in sorted(output_type_dict.items()):
            print(f"\t{k}:\n\t\ttype: {v}")

    main_keys = {
        dataclass_name: _get_dataclass_field_tuples(dataclass_name)
        for k in sorted(all_keys.keys())
        for dataclass_name in [_get_dataclass_name_for_quantity(k)]
        if dataclass_name is not None
    }
    # enumerate every energy model explicitly (see the note above the override)
    if "energy" in all_keys:
        for model in energy_db_models:
            main_keys[model] = _get_dataclass_field_tuples(model)
    output_type_dict = {
        k: v for k, v in sorted(output_type_dict.items(), key=lambda item: item[0])
    }
    return main_keys, output_type_dict


@functools.cache
def _get_quantity_to_dataclass_map() -> Dict[str, str]:
    module = __import__("py4vasp._raw.models", fromlist=[None])
    quantity_to_dataclass: Dict[str, str] = {}
    for class_name in dir(module):
        if not class_name.endswith("Model") or class_name == "_DatabaseModel":
            continue
        base_name = class_name[: -len("Model")]
        quantity_to_dataclass[convert.quantity_name(base_name)] = class_name
    return quantity_to_dataclass


def _get_dataclass_name_for_quantity(quantity_label: str) -> Optional[str]:
    quantity_to_dataclass = _get_quantity_to_dataclass_map()

    direct_match = quantity_to_dataclass.get(quantity_label.lstrip("_"))
    if direct_match is not None:
        return direct_match

    db_key = _quantity_label_to_db_key(quantity_label)
    quantity_part = db_key.split(".", 1)[-1].split(":", 1)[0]
    return quantity_to_dataclass.get(quantity_part.lstrip("_"))


@functools.cache
def _get_dataclass_field_tuples(dataclass_name: str) -> List[Tuple[str, str]]:
    module = __import__("py4vasp._raw.models", fromlist=[None])
    dataclass = getattr(module, dataclass_name, None)
    if dataclass is None:
        return []
    return [
        (field["name"], _format_type_name(field["type"]))
        for field in get_dataclass_fields(dataclass)
    ]


def _format_type_name(field_type: Any) -> str:
    if isinstance(field_type, str):
        return field_type
    field_type_str = str(field_type)
    if field_type_str.startswith("typing."):
        return field_type_str[len("typing.") :]
    if field_type_str.startswith("<class '") and field_type_str.endswith("'>"):
        return field_type_str[len("<class '") : -len("'>")]
    return field_type_str


def _quantity_label_to_db_key(label: str) -> str:
    """Convert a quantity label to a database key format.

    Expected input format:
    <group_name>_<quantity_name>:<selection>

    Returns format:
    <group_name>.<quantity_name>:<selection>
    """
    from py4vasp._calculation import GROUPS, QUANTITIES

    # Split label into group and quantity parts
    for group, quantities in GROUPS.items():
        for quantity in quantities:
            expected_label = f"{group}_{quantity.lstrip('_')}"
            if label.startswith(expected_label):
                split1, split2 = label[: len(group)], label[(len(group) + 1) :]
                if quantity.startswith("_"):
                    split2 = f"_{split2.lstrip('_')}"
                db_key = f"{split1}.{split2}"
                return db_key
    for quantity in QUANTITIES:
        expected_label = quantity.lstrip("_")
        if label.startswith(expected_label):
            db_key = label
            if quantity.startswith("_"):
                db_key = f"_{db_key}"
            return db_key
    return label


def check_schema_snapshot(stored: dict, current: dict) -> Optional[str]:
    """Validate that ``current`` is a legal successor of the committed ``stored`` snapshot.

    Each argument is a schema fingerprint ``{"schema_version": str, "models": {...}}``.
    Returns ``None`` when the transition is allowed, otherwise a human-readable message
    explaining what the developer must do. The rules enforce that any change to the
    models is accompanied by a bump of the schema version:

    - Nothing changed -> OK.
    - Models unchanged but version changed -> rejected (do not bump the version, or
      regenerate the snapshot on a py4vasp release).
    - py4vasp minor series changed -> the ``db`` counter must reset to 1.
    - Same series, models changed -> the ``db`` counter must be strictly greater than
      the stored one.
    """
    s_series, s_counter = parse_schema_version(stored["schema_version"])
    c_series, c_counter = parse_schema_version(current["schema_version"])
    models_equal = stored["models"] == current["models"]
    if c_series != s_series:
        if c_counter != 1:
            return (
                f"The py4vasp version changed the schema series to '{c_series}'; "
                f"reset __DB_SCHEMA__ to 1 in models.py (got {c_counter})."
            )
        return None
    if models_equal:
        if c_counter != s_counter:
            return (
                "The database models are unchanged, so the schema version must stay "
                f"'{stored['schema_version']}' (got '{current['schema_version']}'). "
                "Revert __DB_SCHEMA__ or regenerate the snapshot."
            )
        return None
    if c_counter <= s_counter:
        diff = _format_model_diff(stored["models"], current["models"])
        return (
            "The database models changed. Increment __DB_SCHEMA__ in models.py "
            f"(currently {c_counter}, snapshot recorded {s_counter}) and regenerate "
            "the snapshot with `pytest tests/raw/test_schema_version.py "
            f"--update-schema-snapshot`.\n{diff}"
        )
    return None


def _format_model_diff(old_models: dict, new_models: dict) -> str:
    """Human-readable summary of how two model fingerprints differ."""
    lines: List[str] = []
    added = sorted(set(new_models) - set(old_models))
    removed = sorted(set(old_models) - set(new_models))
    for name in added:
        lines.append(f"+ model {name}")
    for name in removed:
        lines.append(f"- model {name}")
    for name in sorted(set(old_models) & set(new_models)):
        old_fields = {tuple(field) for field in old_models[name]}
        new_fields = {tuple(field) for field in new_models[name]}
        for field in sorted(new_fields - old_fields):
            lines.append(f"+ {name}.{field[0]}: {field[1]}")
        for field in sorted(old_fields - new_fields):
            lines.append(f"- {name}.{field[0]}: {field[1]}")
    return "\n".join(lines)
