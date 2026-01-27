# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import functools
from math import gcd
from typing import Any, Optional, Tuple

from h5py import File

from py4vasp import exception
from py4vasp._raw.data import Version
from py4vasp._raw.definition import DEFAULT_SOURCE, Schema, unique_selections
from py4vasp._raw.schema import Length, Link


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
    tuple[str, str]
        The chemical formula and compound name.
    """
    primitive_numbers = get_primitive_ion_numbers(number_ion_types)
    formula_parts = []
    compound_parts = []
    sorted_types = sorted(zip(ion_types, primitive_numbers), key=lambda x: x[0])
    for ion_type, number in sorted_types:
        if number > 0:
            formula_parts.append(f"{ion_type}{number if number > 1 else ''}")
            compound_parts.append(f"{ion_type}")
    formula = "".join(formula_parts)
    compound = "-".join(compound_parts)
    return formula, compound
