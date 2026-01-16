from typing import Any, Optional, Tuple

from py4vasp import exception
from py4vasp._raw.definition import DEFAULT_SOURCE, unique_selections


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
    has_selection = selection and selection != DEFAULT_SOURCE
    full_key = quantity_name + (f":{selection}" if has_selection else "")
    if group_name is not None:
        full_key = f"{group_name}.{full_key}"
    return full_key, has_selection


def clean_db_key(key: str, db_key_suffix: Optional[str] = None) -> str:
    return (key + (db_key_suffix or "")) if not (":" in key) else key
