from py4vasp._raw.definition import DEFAULT_SOURCE
from py4vasp._util import database


def test_clean_db_key():
    assert (
        database.clean_db_key("group.quantity", db_key_suffix=":selection")
        == "group.quantity:selection"
    )
    assert (
        database.clean_db_key("group.quantity", db_key_suffix=None) == "group.quantity"
    )
    assert (
        database.clean_db_key("group.quantity:selection", db_key_suffix=None)
        == "group.quantity:selection"
    )
    assert (
        database.clean_db_key(
            "group.quantity:selection", db_key_suffix=":other_selection"
        )
        == "group.quantity:selection"
    )


def test_combine_db_dicts():
    dict1 = {
        "a": 1,
        "b": {"c": 2, "d": 3},
        "e": 4,
    }
    dict2 = {
        "b": {"c": 2, "f": 5},
        "g": 6,
    }
    combined = database.combine_db_dicts(dict1, dict2)
    expected = {
        "a": 1,
        "b": {"c": 2, "d": 3, "f": 5},
        "e": 4,
        "g": 6,
    }
    assert combined == expected


def test_construct_database_data_key():
    assert database.construct_database_data_key("group", "quantity", "selection") == (
        "group.quantity:selection",
        True,
    )
    assert database.construct_database_data_key(None, "quantity", None) == (
        "quantity",
        False,
    )
    assert database.construct_database_data_key("group", "quantity", None) == (
        "group.quantity",
        False,
    )
    assert database.construct_database_data_key(None, "quantity", "selection") == (
        "quantity:selection",
        True,
    )
    assert database.construct_database_data_key(
        "group", "quantity", DEFAULT_SOURCE
    ) == ("group.quantity", False)
