# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import ast
import functools
import inspect
from math import gcd
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
    """
    if ion_types is None or number_ion_types is None:
        return None, None, None, None, None

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


def get_all_possible_keys(to_print: bool = False) -> Dict[str, List[str]]:
    """
    Extract all possible keys from _to_database methods in calculation classes.
    
    Returns
    -------
    Dict[str, List[str]]
        A dictionary where keys are the top-level database keys and values are
        lists of nested keys within each top-level key.
    """
    calculation_dir = Path(__file__).parent.parent / "_calculation"
    all_keys = {}
    
    for py_file in calculation_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
            
        try:
            file_keys = _extract_keys_from_file(py_file)
            for key, nested_keys in file_keys.items():
                if key in all_keys:
                    # Merge nested keys, keeping unique ones
                    all_keys[key] = list(set(all_keys[key] + nested_keys))
                else:
                    all_keys[key] = nested_keys
        except Exception as e:
            print(f"Warning: Could not process {py_file.name}: {e}")
            continue
    if to_print:
        for k,v in all_keys.items():
            print(f"{k}: " + ("! EMPTY !" if not v else ""))
            for subkey in v:
                print(f"\t- {subkey}")
    return all_keys


def _extract_keys_from_file(filepath: Path) -> Dict[str, List[str]]:
    """Extract database keys from a single Python file."""
    with open(filepath, 'r') as f:
        tree = ast.parse(f.read(), filename=str(filepath))
    
    keys = {}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_to_database":
            file_keys = _extract_keys_from_function(node, tree)
            keys.update(file_keys)
    
    return keys


def _extract_keys_from_function(func_node: ast.FunctionDef, tree: ast.AST) -> Dict[str, List[str]]:
    """Extract keys from a _to_database function node."""
    keys = {}
    
    # Find all return statements
    for node in ast.walk(func_node):
        if isinstance(node, ast.Return) and node.value:
            if isinstance(node.value, ast.Dict):
                file_keys = _extract_keys_from_dict(node.value, func_node, tree)
                # Only take the first key as per requirements
                if file_keys:
                    first_key = list(file_keys.keys())[0]
                    keys[first_key] = file_keys[first_key]
    
    return keys


def _extract_keys_from_dict(dict_node: ast.Dict, func_node: ast.FunctionDef, tree: ast.AST) -> Dict[str, List[str]]:
    """Extract keys from a dictionary AST node."""
    result = {}
    
    if not dict_node.keys:
        return result
    
    # Get the first key only (as per requirements)
    first_key_node = dict_node.keys[0]
    first_value_node = dict_node.values[0]
    
    if isinstance(first_key_node, ast.Constant):
        first_key = first_key_node.value
        nested_keys = _extract_nested_keys(first_value_node, func_node, tree)
        result[first_key] = nested_keys
    
    return result


def _extract_nested_keys(value_node: ast.AST, func_node: ast.FunctionDef, tree: ast.AST) -> List[str]:
    """Extract nested keys from a value node."""
    keys = []
    
    if isinstance(value_node, ast.Dict):
        for key_node in value_node.keys:
            if isinstance(key_node, ast.Constant):
                keys.append(key_node.value)
            elif key_node is None:
                # This is a **dict unpacking
                pass
        
        # Handle **dict unpacking
        for val_node in value_node.values:
            if isinstance(val_node, ast.Call):
                # Handle function calls like self._dict_from_runtime()
                unpacked_keys = _extract_keys_from_method_call(val_node, func_node, tree)
                keys.extend(unpacked_keys)
    
    return keys


def _extract_keys_from_method_call(call_node: ast.Call, func_node: ast.FunctionDef, tree: ast.AST) -> List[str]:
    """Extract keys from a method call that returns a dictionary."""
    keys = []
    
    # Get the method name
    if isinstance(call_node.func, ast.Attribute):
        method_name = call_node.func.attr
        
        # Find the method definition in the same class
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == method_name:
                # Find return statements in this method
                for ret_node in ast.walk(node):
                    if isinstance(ret_node, ast.Return) and ret_node.value:
                        if isinstance(ret_node.value, ast.Dict):
                            for key_node in ret_node.value.keys:
                                if isinstance(key_node, ast.Constant):
                                    keys.append(key_node.value)
    
    return keys
