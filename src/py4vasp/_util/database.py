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
    try:
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
    except Exception:
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
                    - A dictionary where keys are database dataclass names (e.g. Band_DB) and
                        values are lists of tuples containing (attribute_name, attribute_type).
                    - A dictionary mapping all available keys (group.quantity[:selection]) to
                        their dataclass names. If no matching dataclass exists, the value is None.
                        The default selection is represented without a suffix, e.g. "band"
                        instead of "band:default".
    """
    calculation_dir = Path(__file__).parent.parent / "_calculation"
    all_keys = {}

    from py4vasp._calculation import GROUPS, QUANTITIES

    _USE_LEGACY = False

    for py_file in calculation_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue

        try:
            file_keys, classes_without_method = _extract_keys_from_file(
                py_file, debug, _USE_LEGACY
            )

            if debug:
                print(f"\n=== DEBUG {py_file.name} ===")
                print(f"file_keys: {file_keys}")

            if (len(file_keys) == 0) and (len(classes_without_method) == 0):
                all_keys[py_file.stem] = []

            for key, nested_keys in file_keys.items():
                if (
                    (key in QUANTITIES)
                    or (f"_{key}" in QUANTITIES)
                    or any([key.startswith(f"{group}_") for group in GROUPS.keys()])
                ):
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
                if (
                    (class_name in QUANTITIES)
                    or (f"_{class_name}" in QUANTITIES)
                    or any(
                        [class_name.startswith(f"{group}_") for group in GROUPS.keys()]
                    )
                ):
                    if class_name not in all_keys:
                        all_keys[class_name] = None

        except Exception as e:
            print(f"Warning: Could not process {py_file.name}: {e}")
            if debug:
                import traceback

                traceback.print_exc()
            continue

    output_type_dict: dict[str, Optional[str]] = {}
    """Map all available keys (group.quantity[:selection]) to dataclass names."""

    for k in list(all_keys.keys()):
        try:
            selections_list = unique_selections(k)
        except:
            selections_list = []
        constructed_key = _quantity_label_to_db_key(k)
        dataclass_name = _get_dataclass_name_for_quantity(k)
        for sel in selections_list:
            if sel in GROUPS.keys():
                output_type_dict[f"{sel}._{k}"] = dataclass_name
        if len(selections_list) == 0:
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
    output_type_dict = {
        k: v for k, v in sorted(output_type_dict.items(), key=lambda item: item[0])
    }
    return main_keys, output_type_dict


@functools.cache
def _get_quantity_to_dataclass_map() -> Dict[str, str]:
    module = __import__("py4vasp._raw.data_db", fromlist=[None])
    quantity_to_dataclass: Dict[str, str] = {}
    for class_name in dir(module):
        if not class_name.endswith("_DB"):
            continue
        base_name = class_name[: -len("_DB")]
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
    module = __import__("py4vasp._raw.data_db", fromlist=[None])
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


def _get_constant_value(node: ast.AST) -> Optional[str]:
    """Extract constant string value from AST node."""
    if isinstance(node, ast.Constant):
        return node.value
    return None


def _extract_keys_from_file(
    filepath: Path, debug: bool = False, use_legacy: bool = False
) -> tuple[Dict[str, List[str]], List[str]]:
    """Extract database keys from a single Python file."""
    with open(filepath, "r") as f:
        content = f.read()
        tree = ast.parse(content, filename=str(filepath))

    keys = {}
    refinery_classes = set()
    classes_with_method = set()

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
                            if debug:
                                print(f"\n[{filepath.stem}] Found _to_database method")
                            if use_legacy:
                                file_keys = _extract_keys_from_function(
                                    item, tree, debug
                                )
                            else:
                                file_keys = _extract_keys_from_dataclass(node.name)
                            if debug:
                                print(f"[{filepath.stem}] Result: {file_keys}")
                            keys.update(file_keys)

    classes_without_method = list(refinery_classes - classes_with_method)
    return keys, classes_without_method


def _extract_keys_from_dataclass(class_key: str) -> Dict[str, List[str]]:
    """Extract keys from a dataclass-based _to_database method."""
    # This is a simplified version that assumes the dataclass fields correspond to the keys.
    # It does not handle complex logic, but it should work for straightforward cases.
    try:
        module = __import__(f"py4vasp._raw.data_db", fromlist=[None])
        cls = getattr(module, f"{class_key}_DB")
        from dataclasses import fields

        return {convert.quantity_name(class_key): [field.name for field in fields(cls)]}
    except Exception as e:
        print(f"Error extracting keys from dataclass {class_key}: {e}")
    return {}


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
            file_keys = _resolve_return(stmt.value, {}, func_node, tree, debug)
            if file_keys:
                return file_keys

    return {}


def _extract_dict_structure(
    dict_node: ast.Dict,
    func_node: ast.FunctionDef,
    tree: ast.AST,
    intermediate_dicts: dict = None,
) -> Dict[str, List[str]]:
    """Extract first key and all nested keys from a dict node.

    Handles **unpacking, method calls, and variable references.
    """
    if intermediate_dicts is None:
        intermediate_dicts = {}

    result = {}
    if not dict_node.keys:
        return result

    # Get first key
    first_key = _get_constant_value(dict_node.keys[0])
    if not first_key:
        return result

    # Extract nested keys from the first value
    first_value = dict_node.values[0]

    if isinstance(first_value, ast.Dict):
        # Nested dict - extract all its keys
        nested_keys = _extract_all_dict_keys(
            first_value, func_node, tree, intermediate_dicts
        )
    elif isinstance(first_value, ast.Call):
        # Method call - resolve it
        nested_keys = _resolve_call(first_value, func_node, tree)
    elif isinstance(first_value, ast.Name) and first_value.id in intermediate_dicts:
        # Variable reference
        nested_keys = intermediate_dicts[first_value.id]
    else:
        nested_keys = []

    result[first_key] = nested_keys
    return result


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
                    intermediate_dicts[var_name] = _extract_all_dict_keys(
                        stmt.value, func_node, tree, intermediate_dicts
                    )
                    intermediate_dicts[f"_ast_{var_name}"] = stmt.value
                # Track dictionary comprehensions
                elif isinstance(stmt.value, ast.DictComp):
                    comp_keys = _extract_keys_from_dict_comp(stmt.value, tree)
                    if comp_keys:
                        intermediate_dicts[var_name] = comp_keys
                # Track method calls
                elif isinstance(stmt.value, ast.Call):
                    method_keys = _resolve_call(stmt.value, func_node, tree)
                    if method_keys:
                        intermediate_dicts[var_name] = method_keys

        # Check for loop-based dict construction
        elif isinstance(stmt, ast.For):
            _track_dict_construction_in_loop(stmt, intermediate_dicts, func_node, tree)

        # Find return statement
        if isinstance(stmt, ast.Return) and stmt.value:
            result = _resolve_return(
                stmt.value, intermediate_dicts, func_node, tree, debug
            )
            if result:
                return result

    return result


def _get_global_var_items(
    var_name: str, tree: ast.AST, use_dict_values: bool = False
) -> List[str]:
    """Extract items from a global variable (dict, tuple, or list).

    Parameters
    ----------
    var_name : str
        Name of the global variable
    tree : ast.AST
        AST tree to search
    use_dict_values : bool
        For dicts, extract values instead of keys (for .items() loops)
    """
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == var_name:
                    value = node.value

                    # Handle dict
                    if isinstance(value, ast.Dict):
                        nodes = value.values if use_dict_values else value.keys
                        return [
                            _get_constant_value(n)
                            for n in nodes
                            if _get_constant_value(n)
                        ]

                    # Handle tuple/list
                    elif isinstance(value, (ast.Tuple, ast.List)):
                        return [
                            _get_constant_value(elt)
                            for elt in value.elts
                            if _get_constant_value(elt)
                        ]
    return []


def _extract_keys_from_dict_comp(comp_node: ast.DictComp, tree: ast.AST) -> List[str]:
    """Extract keys from a dictionary comprehension."""
    keys = []
    key_node = comp_node.key

    for generator in comp_node.generators:
        iter_node = generator.iter
        iter_values = []

        if isinstance(iter_node, ast.Name):
            iter_values = _get_global_var_items(iter_node.id, tree)
        elif isinstance(iter_node, (ast.Tuple, ast.List)):
            iter_values = [
                _get_constant_value(elt)
                for elt in iter_node.elts
                if _get_constant_value(elt)
            ]

        # Generate keys from pattern
        if isinstance(key_node, ast.JoinedStr):
            for value in iter_values:
                key = _evaluate_fstring(key_node, generator.target, value)
                if key:
                    keys.append(key)
        elif key_node:
            static_key = _get_constant_value(key_node)
            if static_key:
                keys.append(static_key)

    return keys


def _evaluate_fstring(
    fstring_node: ast.JoinedStr, target_node: ast.AST, substitution_value: str
) -> Optional[str]:
    """Evaluate f-string by substituting iteration variable with a concrete value."""
    target_name = target_node.id if isinstance(target_node, ast.Name) else None
    if not target_name:
        return None

    result_parts = []
    for part in fstring_node.values:
        # Handle constant string parts
        const_val = _get_constant_value(part)
        if const_val:
            result_parts.append(const_val)
        # Handle variable substitution
        elif isinstance(part, ast.FormattedValue):
            if isinstance(part.value, ast.Name) and part.value.id == target_name:
                result_parts.append(substitution_value)
            else:
                return None  # Can't evaluate complex expressions

    return "".join(result_parts)


def _resolve_return(
    return_node: ast.AST,
    intermediate_dicts: dict,
    func_node: ast.FunctionDef,
    tree: ast.AST,
    debug: bool = False,
) -> Dict[str, List[str]]:
    """Resolve a return statement to extract database keys."""
    # Resolve variable references to their stored AST nodes or cached keys
    if isinstance(return_node, ast.Name):
        var_name = return_node.id
        ast_key = f"_ast_{var_name}"
        if ast_key in intermediate_dicts:
            # Use the stored AST node for full structural analysis
            return_node = intermediate_dicts[ast_key]
        elif var_name in intermediate_dicts:
            # Return already-extracted keys (simple case)
            return {var_name: intermediate_dicts[var_name]}
        else:
            return {}

    # Handle dict literal returns
    if isinstance(return_node, ast.Dict):
        return _extract_dict_structure(return_node, func_node, tree, intermediate_dicts)

    # Handle function call returns (e.g., combine_db_dicts)
    if isinstance(return_node, ast.Call):
        return _resolve_call_return(
            return_node, func_node, tree, intermediate_dicts, debug
        )

    return {}


def _resolve_call(
    call_node: ast.Call, func_node: ast.FunctionDef, tree: ast.AST
) -> List[str]:
    """Resolve a method/function call to extract returned dict keys."""
    method_name = None
    if isinstance(call_node.func, ast.Attribute):
        method_name = call_node.func.attr
    elif isinstance(call_node.func, ast.Name):
        method_name = call_node.func.id

    if not method_name:
        return []

    # Find method definition
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for class_item in node.body:
                if (
                    isinstance(class_item, ast.FunctionDef)
                    and class_item.name == method_name
                ):
                    # Look for dict return
                    for stmt in class_item.body:
                        if isinstance(stmt, ast.Return) and isinstance(
                            stmt.value, ast.Dict
                        ):
                            return _extract_all_dict_keys(
                                stmt.value, class_item, tree, {}
                            )
    return []


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
                                    source_keys = _get_global_var_items(
                                        source_dict, tree, use_dict_values=True
                                    )
                                    if source_keys:
                                        transformed_keys = _transform_keys_from_loop(
                                            source_keys, target.slice
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
                                        )
                                        if dict_name not in intermediate_dicts:
                                            intermediate_dicts[dict_name] = []
                                        intermediate_dicts[dict_name].extend(
                                            transformed_keys
                                        )


def _transform_keys_from_loop(source_keys: List[str], slice_node: ast.AST) -> List[str]:
    """Transform keys based on f-string or concatenation pattern."""
    if not isinstance(slice_node, ast.JoinedStr):
        return source_keys

    # Extract suffix patterns from f-string
    suffixes = []
    has_variable = False
    for part in slice_node.values:
        const_val = _get_constant_value(part)
        if const_val:
            suffixes.append(const_val)
        elif isinstance(part, ast.FormattedValue):
            has_variable = True

    # If there's a variable in the f-string, apply suffix to each source key
    if has_variable and suffixes:
        suffix = "".join(suffixes)
        return [f"{key}{suffix}" for key in source_keys]

    return source_keys


def _extract_all_dict_keys(
    dict_node: ast.Dict,
    func_node: ast.FunctionDef,
    tree: ast.AST,
    intermediate_dicts: dict = None,
) -> List[str]:
    """Extract all keys from a dict, handling literal keys and **unpacked dicts."""
    if intermediate_dicts is None:
        intermediate_dicts = {}

    keys = []
    for key_node, val_node in zip(dict_node.keys, dict_node.values):
        if key_node is None:
            # **dict unpacking - resolve the unpacked dictionary
            if isinstance(val_node, ast.Call):
                keys.extend(_resolve_call(val_node, func_node, tree))
            elif isinstance(val_node, ast.Name) and val_node.id in intermediate_dicts:
                keys.extend(intermediate_dicts[val_node.id])
        else:
            # Literal key
            const_key = _get_constant_value(key_node)
            if const_key:
                keys.append(const_key)

    return keys


def _resolve_call_return(
    call_node: ast.Call,
    func_node: ast.FunctionDef,
    tree: ast.AST,
    intermediate_dicts: dict = None,
    debug: bool = False,
) -> Dict[str, List[str]]:
    """Extract keys from a function call in a return statement.

    Handles special cases like combine_db_dicts where we need to analyze arguments.
    """
    if intermediate_dicts is None:
        intermediate_dicts = {}

    if not call_node.args:
        return {}

    # Get function name
    func_name = None
    if isinstance(call_node.func, ast.Name):
        func_name = call_node.func.id
    elif isinstance(call_node.func, ast.Attribute):
        func_name = call_node.func.attr

    first_arg = call_node.args[0]

    # Handle combine_db_dicts - extract from first argument
    if func_name and "combine" in func_name.lower():
        if isinstance(first_arg, ast.Dict):
            return _extract_dict_structure(
                first_arg, func_node, tree, intermediate_dicts
            )
        elif isinstance(first_arg, ast.Name):
            return _resolve_return(
                first_arg, intermediate_dicts, func_node, tree, debug
            )

    # Fallback: try to extract from first dict argument
    if isinstance(first_arg, ast.Dict):
        return _extract_dict_structure(first_arg, func_node, tree, intermediate_dicts)

    return {}
