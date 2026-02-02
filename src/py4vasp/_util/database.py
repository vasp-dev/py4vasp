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

    from py4vasp._calculation import GROUPS, QUANTITIES

    # Find private quantities
    private_quantities = [
        (None, quantity) for quantity in QUANTITIES if quantity.startswith("_")
    ] + [
        (group, quantity)
        for group, quantities in GROUPS.items()
        for quantity in quantities
        if quantity.startswith("_")
    ]

    # Fix keys to change private quantity keys back to private
    relevant_keys = []
    for group, quantity in private_quantities:
        if group is None:
            expected_key = quantity.lstrip("_")
        else:
            expected_key = f"{group}.{quantity.lstrip('_')}"
        relevant_keys = relevant_keys + [
            key
            for key in dict_to_clean.keys()
            if key.startswith(f"{expected_key}:") or key == expected_key
        ]
    relevant_keys = set(relevant_keys)
    for key in relevant_keys:
        if key in dict_to_clean:
            dict_to_clean[f"_{key}"] = dict_to_clean.pop(key)

    # Fix keys to resolve group selections
    relevant_keys = []
    for group, _ in GROUPS.items():
        expected_key = group
        relevant_keys = [
            key for key in dict_to_clean.keys() if key.endswith(f":{group}")
        ]
        for key in relevant_keys:
            split1, split2 = key.rsplit(":", 1)
            dict_to_clean[f"{split2}._{split1.lstrip('_')}"] = dict_to_clean.pop(key)

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


def get_all_possible_keys(
    to_print: bool = False, debug: bool = False
) -> Dict[str, List[str]]:
    """
    Extract all possible keys from _to_database methods in calculation classes.

    Returns
    -------
    Dict[str, List[str]]
        A dictionary where keys are the top-level database keys and values are
        lists of nested keys within each top-level key. If a class inherits from
        base.Refinery but doesn't implement _to_database, the value is None.
    """
    calculation_dir = Path(__file__).parent.parent / "_calculation"
    all_keys = {}

    for py_file in calculation_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue

        try:
            file_keys, classes_without_method = _extract_keys_from_file(py_file, debug)

            if debug:
                print(f"\n=== DEBUG {py_file.name} ===")
                print(f"file_keys: {file_keys}")

            if (len(file_keys) == 0) and (len(classes_without_method) == 0):
                all_keys[py_file.stem] = []

            for key, nested_keys in file_keys.items():
                if key in all_keys:
                    if all_keys[key] is not None and nested_keys is not None:
                        # Merge nested keys, keeping unique ones
                        all_keys[key] = list(set(all_keys[key] + nested_keys))
                    elif nested_keys is not None:
                        all_keys[key] = nested_keys
                else:
                    all_keys[key] = nested_keys

            # Add classes without _to_database method
            for class_name in classes_without_method:
                if class_name not in all_keys:
                    all_keys[class_name] = None

        except Exception as e:
            print(f"Warning: Could not process {py_file.name}: {e}")
            if debug:
                import traceback

                traceback.print_exc()
            continue

    for k in list(all_keys.keys()):
        selections = _get_unique_selections_str(k)
        if selections == "NONE":
            all_keys.pop(k)

    if to_print:
        print("\n--- PARSED KEYS: ---")
        for k, v in sorted(all_keys.items()):
            if v is not None and len(v) > 0:
                print(
                    f"\t{_quantity_label_to_db_key(k)}:("
                    + _get_unique_selections_str(k)
                    + ")"
                )
                should_sort = k in ["energy"]
                vsort = sorted(v) if should_sort else v
                for subkey in vsort:
                    print(f"\t\t- {subkey}")

        print("\n--- EMPTY KEYS ---")
        for k, v in sorted(all_keys.items()):
            if v is not None and len(v) == 0:
                print(
                    f"\t{_quantity_label_to_db_key(k)}:("
                    + _get_unique_selections_str(k)
                    + ")"
                )

        print("\n--- MISSING _to_database ---")
        for k, v in sorted(all_keys.items()):
            if v is None:
                print(
                    f"\t{_quantity_label_to_db_key(k)}:("
                    + _get_unique_selections_str(k)
                    + ")"
                )

    # TODO fix remaining EMPTY KEYS where they should not be empty
    # TODO discuss with Martin and/or Zahed whether the keys in the dictionary should be group.quantity:selection or quantity:selection
    # (then selection can also be a group) -- decide this for get_all_possible_keys and available_quantities, then fix accordingly

    for k in list(all_keys.keys()):
        new_label = _quantity_label_to_db_key(k)
        if new_label != k:
            all_keys[new_label] = all_keys.pop(k)

    all_keys = {k: v for k, v in sorted(all_keys.items(), key=lambda item: item[0])}
    return all_keys


def _get_unique_selections_str(key: str) -> str:
    """Get a string representation of unique selections for a given key."""
    selections = []
    try:
        selections = unique_selections(key)
        return ", ".join(selections)
    except:
        return "NONE"


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
                    split2 = f"_{split2}"
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


def _extract_keys_from_file(
    filepath: Path, debug: bool = False
) -> tuple[Dict[str, List[str]], List[str]]:
    """Extract database keys from a single Python file."""
    with open(filepath, "r") as f:
        content = f.read()
        tree = ast.parse(content, filename=str(filepath))

    keys = {}
    refinery_classes = set()
    classes_with_method = set()

    is_debug_file = debug and filepath.stem in ["run_info", "energy", "phonon_band"]

    # First pass: find all Refinery classes
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            # Check if it inherits from base.Refinery or Refinery
            for base in node.bases:
                base_name = ""
                if isinstance(base, ast.Attribute):
                    base_name = (
                        f"{base.value.id}.{base.attr}"
                        if isinstance(base.value, ast.Name)
                        else ""
                    )
                elif isinstance(base, ast.Name):
                    base_name = base.id

                if "Refinery" in base_name:
                    # Convert class name to snake_case for database key
                    class_key = convert._to_snakecase(node.name)
                    refinery_classes.add(class_key)

                    # Look for _to_database method
                    for item in node.body:
                        if (
                            isinstance(item, ast.FunctionDef)
                            and item.name == "_to_database"
                        ):
                            classes_with_method.add(class_key)
                            if is_debug_file:
                                print(f"\n[{filepath.stem}] Found _to_database method")
                            file_keys = _extract_keys_from_function(
                                item, tree, is_debug_file
                            )
                            if is_debug_file:
                                print(f"[{filepath.stem}] Result: {file_keys}")
                            keys.update(file_keys)

    classes_without_method = list(refinery_classes - classes_with_method)
    return keys, classes_without_method


def _extract_keys_from_function(
    func_node: ast.FunctionDef, tree: ast.AST, debug: bool = False
) -> Dict[str, List[str]]:
    """Extract keys from a _to_database function node."""
    if debug:
        print(f"[FUNC] Analyzing _to_database")
    # First try the comprehensive body analysis (handles complex cases)
    file_keys = _extract_keys_from_body(func_node, tree, debug)
    if file_keys:
        return file_keys

    # Fallback: look for simple direct returns
    for stmt in func_node.body:
        if isinstance(stmt, ast.Return) and stmt.value:
            if isinstance(stmt.value, ast.Dict):
                file_keys = _extract_keys_from_dict(stmt.value, func_node, tree, debug)
                if file_keys:
                    return file_keys
            elif isinstance(stmt.value, ast.Call):
                file_keys = _extract_keys_from_call_return(
                    stmt.value, func_node, tree, {}, debug
                )
                if file_keys:
                    return file_keys

    return {}


def _extract_keys_from_dict(
    dict_node: ast.Dict, func_node: ast.FunctionDef, tree: ast.AST
) -> Dict[str, List[str]]:
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
    elif isinstance(first_key_node, ast.Str):
        first_key = first_key_node.s
        nested_keys = _extract_nested_keys(first_value_node, func_node, tree)
        result[first_key] = nested_keys

    return result


def _extract_nested_keys(
    value_node: ast.AST, func_node: ast.FunctionDef, tree: ast.AST
) -> List[str]:
    """Extract nested keys from a value node."""
    keys = []

    if isinstance(value_node, ast.Dict):
        # Collect all keys including from **unpacking
        for i, key_node in enumerate(value_node.keys):
            if key_node is None:
                # **dict unpacking
                val_node = value_node.values[i]
                if isinstance(val_node, ast.Call):
                    unpacked_keys = _extract_keys_from_method_call(
                        val_node, func_node, tree
                    )
                    keys.extend(unpacked_keys)
            elif isinstance(key_node, ast.Constant):
                keys.append(key_node.value)
            elif isinstance(key_node, ast.Str):
                keys.append(key_node.s)
    elif isinstance(value_node, ast.Call):
        # The entire value is a method call
        keys = _extract_keys_from_method_call(value_node, func_node, tree)

    return keys


def _extract_keys_from_body(
    func_node: ast.FunctionDef, tree: ast.AST, debug: bool = False
) -> Dict[str, List[str]]:
    """Analyze entire function body to extract keys."""
    result = {}
    intermediate_dicts = {}

    if debug:
        print(f"[BODY] Analyzing function body with {len(func_node.body)} statements")

    for stmt in func_node.body:
        # Track variable assignments
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if isinstance(target, ast.Name):
                var_name = target.id

                # Track dict literals
                if isinstance(stmt.value, ast.Dict):
                    intermediate_dicts[var_name] = _get_all_dict_keys(
                        stmt.value, func_node, tree, intermediate_dicts
                    )
                # Track method calls that return dicts
                elif isinstance(stmt.value, ast.Call):
                    method_keys = _extract_keys_from_method_call(
                        stmt.value, func_node, tree
                    )
                    if method_keys:
                        intermediate_dicts[var_name] = method_keys

        # Check for loop-based dict construction
        elif isinstance(stmt, ast.For):
            _track_dict_construction_in_loop(stmt, intermediate_dicts, func_node, tree)

        # Find return statement
        if isinstance(stmt, ast.Return) and stmt.value:
            if debug:
                print(f"[BODY] Found return statement")
            if isinstance(stmt.value, ast.Dict):
                if debug:
                    print(f"[BODY] Return value is a Dict")
                result = _resolve_return_dict(
                    stmt.value, intermediate_dicts, func_node, tree, debug
                )
                if result:
                    return result
            elif isinstance(stmt.value, ast.Call):
                if debug:
                    print(f"[BODY] Return value is a Call")
                result = _extract_keys_from_call_return(
                    stmt.value, func_node, tree, intermediate_dicts, debug
                )
                if result:
                    return result

    return result


def _resolve_return_dict(
    dict_node: ast.Dict,
    intermediate_dicts: dict,
    func_node: ast.FunctionDef,
    tree: ast.AST,
    debug: bool = False,
) -> Dict[str, List[str]]:
    """Resolve the final return dictionary."""
    result = {}

    if not dict_node.keys:
        return result

    # Get first key
    first_key_node = dict_node.keys[0]
    first_value_node = dict_node.values[0]

    if not isinstance(first_key_node, (ast.Constant, ast.Str)):
        return result

    first_key = (
        first_key_node.value
        if isinstance(first_key_node, ast.Constant)
        else first_key_node.s
    )

    if debug:
        print(f"[RESOLVE] First key: {first_key}")
        print(f"[RESOLVE] Value node type: {type(first_value_node).__name__}")

    # Check if value is a method call (like self.to_dict())
    if isinstance(first_value_node, ast.Call):
        if debug:
            print(f"[RESOLVE] Calling _extract_keys_from_method_call")
        method_keys = _extract_keys_from_method_call(
            first_value_node, func_node, tree, debug
        )
        if debug:
            print(f"[RESOLVE] Got keys: {method_keys}")
        if method_keys:
            result[first_key] = method_keys
            return result

    # Check if value references an intermediate dict variable
    if isinstance(first_value_node, ast.Name):
        var_name = first_value_node.id
        if var_name in intermediate_dicts:
            result[first_key] = intermediate_dicts[var_name]
            return result

    # Check if value is a dict literal (possibly with unpacking)
    if isinstance(first_value_node, ast.Dict):
        nested_keys = _get_all_dict_keys(
            first_value_node, func_node, tree, intermediate_dicts
        )
        result[first_key] = nested_keys
        return result

    return result


def _extract_keys_from_method_call(
    call_node: ast.Call, func_node: ast.FunctionDef, tree: ast.AST, debug: bool = False
) -> List[str]:
    """Extract keys from a method call that returns a dictionary."""
    keys = []

    # Get the method name
    method_name = None
    if isinstance(call_node.func, ast.Attribute):
        method_name = call_node.func.attr
    elif isinstance(call_node.func, ast.Name):
        method_name = call_node.func.id

    if method_name is None:
        return keys

    if debug:
        print(f"[METHOD] Looking for method: {method_name}")

    # Find the method definition in the same file
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for class_item in node.body:
                if (
                    isinstance(class_item, ast.FunctionDef)
                    and class_item.name == method_name
                ):
                    if debug:
                        print(f"[METHOD] Found {method_name}")
                    # Look for return statements in this method
                    for stmt in class_item.body:
                        if isinstance(stmt, ast.Return) and stmt.value:
                            if isinstance(stmt.value, ast.Dict):
                                if debug:
                                    print(f"[METHOD] {method_name} returns a Dict")
                                # Use _get_all_dict_keys to handle **unpacking
                                all_keys = _get_all_dict_keys(
                                    stmt.value, class_item, tree, {}, debug
                                )
                                if debug:
                                    print(f"[METHOD] Extracted keys: {all_keys}")
                                return all_keys

    return keys


def _track_dict_construction_in_loop(
    for_stmt: ast.For,
    intermediate_dicts: dict,
    func_node: ast.FunctionDef,
    tree: ast.AST,
):
    """Track dictionaries being built in loops."""
    # Find dict being populated in loop body
    for stmt in for_stmt.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Subscript):
                    if isinstance(target.value, ast.Name):
                        dict_name = target.value.id

                        # Try to infer keys from the loop source
                        if isinstance(for_stmt.iter, ast.Call):
                            if (
                                isinstance(for_stmt.iter.func, ast.Attribute)
                                and for_stmt.iter.func.attr == "items"
                            ):
                                # Iterating over another dict's items
                                if isinstance(for_stmt.iter.func.value, ast.Name):
                                    source_dict = for_stmt.iter.func.value.id

                                    # Check if iterating over a known global dict (like _DB_KEYS)
                                    # When iterating over .items(), we want the values, not keys
                                    source_keys = _get_keys_from_global_dict(
                                        source_dict, tree, use_values=True
                                    )
                                    if source_keys:
                                        transformed_keys = _transform_keys_from_loop(
                                            source_keys,
                                            target.slice,
                                            for_stmt,
                                        )
                                        if dict_name not in intermediate_dicts:
                                            intermediate_dicts[dict_name] = []
                                        intermediate_dicts[dict_name].extend(
                                            transformed_keys
                                        )
                                    elif source_dict in intermediate_dicts:
                                        # Apply transformation based on the key pattern
                                        transformed_keys = _transform_keys_from_loop(
                                            intermediate_dicts[source_dict],
                                            target.slice,
                                            for_stmt,
                                        )
                                        if dict_name not in intermediate_dicts:
                                            intermediate_dicts[dict_name] = []
                                        intermediate_dicts[dict_name].extend(
                                            transformed_keys
                                        )


def _get_keys_from_global_dict(
    dict_name: str, tree: ast.AST, use_values: bool = False
) -> List[str]:
    """Extract keys or values from a global dictionary definition in the file.

    Parameters
    ----------
    dict_name : str
        Name of the dictionary variable to find
    tree : ast.AST
        The AST tree to search
    use_values : bool
        If True, extract dictionary values instead of keys (useful for .items() loops)
    """
    # Look for module-level assignments like _DB_KEYS = {...}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == dict_name:
                    if isinstance(node.value, ast.Dict):
                        items = []
                        nodes_to_extract = (
                            node.value.values if use_values else node.value.keys
                        )
                        for item_node in nodes_to_extract:
                            if isinstance(item_node, ast.Constant):
                                items.append(item_node.value)
                            elif isinstance(item_node, ast.Str):
                                items.append(item_node.s)
                        return items
    return []


def _transform_keys_from_loop(
    source_keys: List[str], slice_node: ast.AST, for_node: ast.For
) -> List[str]:
    """Transform keys based on f-string or concatenation pattern."""
    transformed = []

    # Check if using f-string
    if isinstance(slice_node, ast.JoinedStr):
        # Extract suffix patterns
        suffixes = []
        has_variable = False
        for part in slice_node.values:
            if isinstance(part, ast.Constant):
                suffixes.append(part.value)
            elif isinstance(part, ast.Str):
                suffixes.append(part.s)
            elif isinstance(part, ast.FormattedValue):
                has_variable = True

        # If there's a variable in the f-string, assume it's iterating source keys
        if has_variable and suffixes:
            # Apply suffix to each source key
            suffix = "".join(suffixes)
            for key in source_keys:
                transformed.append(f"{key}{suffix}")
        else:
            transformed = source_keys
    else:
        transformed = source_keys

    return transformed


def _get_all_dict_keys(
    dict_node: ast.Dict,
    func_node: ast.FunctionDef,
    tree: ast.AST,
    intermediate_dicts: dict = None,
    debug: bool = False,
) -> List[str]:
    """Get all keys from a dict node, including unpacked dicts."""
    if intermediate_dicts is None:
        intermediate_dicts = {}

    keys = []

    for i, (key_node, val_node) in enumerate(zip(dict_node.keys, dict_node.values)):
        if key_node is None:
            # **dict unpacking
            if debug:
                print(f"[UNPACK] Found ** unpacking")
            if isinstance(val_node, ast.Call):
                unpacked = _extract_keys_from_method_call(
                    val_node, func_node, tree, debug
                )
                if debug:
                    print(f"[UNPACK] Got keys: {unpacked}")
                keys.extend(unpacked)
            elif isinstance(val_node, ast.Name):
                # Variable reference
                if intermediate_dicts and val_node.id in intermediate_dicts:
                    keys.extend(intermediate_dicts[val_node.id])
        elif isinstance(key_node, ast.Constant):
            keys.append(key_node.value)
        elif isinstance(key_node, ast.Str):
            keys.append(key_node.s)

    return keys


def _extract_keys_from_call_return(
    call_node: ast.Call,
    func_node: ast.FunctionDef,
    tree: ast.AST,
    intermediate_dicts: dict = None,
    debug: bool = False,
) -> Dict[str, List[str]]:
    """Extract keys from a return statement that calls a function."""
    if intermediate_dicts is None:
        intermediate_dicts = {}

    # Handle combine_db_dicts or similar functions
    func_name = None
    if isinstance(call_node.func, ast.Name):
        func_name = call_node.func.id
    elif isinstance(call_node.func, ast.Attribute):
        func_name = call_node.func.attr

    # If it's combine_db_dicts, look at the first argument
    if func_name and "combine" in func_name.lower():
        if call_node.args:
            first_arg = call_node.args[0]
            if isinstance(first_arg, ast.Dict):
                # Process the first dict argument
                result = {}
                if first_arg.keys and len(first_arg.keys) > 0:
                    first_key_node = first_arg.keys[0]
                    first_value_node = first_arg.values[0]

                    if isinstance(first_key_node, (ast.Constant, ast.Str)):
                        first_key = (
                            first_key_node.value
                            if isinstance(first_key_node, ast.Constant)
                            else first_key_node.s
                        )

                        # Get all nested keys including from **unpacking
                        nested_keys = _get_all_dict_keys(
                            (
                                first_value_node
                                if isinstance(first_value_node, ast.Dict)
                                else first_arg
                            ),
                            func_node,
                            tree,
                            intermediate_dicts,
                        )
                        result[first_key] = nested_keys
                        return result
            elif isinstance(first_arg, ast.Name):
                # Variable reference
                if first_arg.id in intermediate_dicts:
                    # Need to reconstruct the top-level key
                    # This is tricky - we need more context
                    pass

    # Fallback
    if call_node.args:
        first_arg = call_node.args[0]
        if isinstance(first_arg, ast.Dict):
            return _extract_keys_from_dict(first_arg, func_node, tree)

    return {}
