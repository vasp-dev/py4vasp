def no_return_info(a: int, b: str) -> None:
    """Function without return type information in the docstring.

    Parameters
    ----------
    a : int
        An integer parameter.
    b : str
        A string parameter.
    """
    return None


def no_return_description(a: int, b: str):
    """Function with return type in signature but no Returns field.

    Returns
    -------
    int
    """
    return a + len(b)


def return_with_description(a: int, b: str):
    """Function with return type and description in Returns field.

    Returns
    -------
    int
        The sum of the integer and the length of the string.
    """
    return a + len(b)


def return_with_type_only(a: int, b: str):
    """Function with return type only in Returns field.

    Returns
    -------
    int
    """
    return a + len(b)


def return_with_description_only(a: int, b: str) -> int:
    """Function with return description only in Returns field.

    Returns
    -------
    The sum of the integer and the length of the string.
    """
    return a + len(b)


def return_with_no_info(a: int, b: str):
    """Function with no return type information.

    Parameters
    ----------
    a : int
        An integer parameter.
    b : str
        A string parameter.

    Returns
    -------
    Returns a multiple of a.
    """
    return a * 2


def return_with_multiline_type_description_no_type(a: int, b: str):
    """Function with type description across multiple lines and no type given.

    Returns
    -------
    This integer is returned
    and represents the sum of a
    and length of b.
    """
    return a + len(b)


def return_with_multiline_type_description_field_type(a: int, b: str):
    """Function with type description across multiple lines and type given in Returns field.

    Returns
    -------
    int
        This integer is returned
        and represents the sum of a
        and length of b.
    """
    return a + len(b)


def return_with_multiline_type_description_sig_type(a: int, b: str) -> int:
    """Function with type description across multiple lines and type given in signature.

    Returns
    -------
    This integer is returned
    and represents the sum of a
    and length of b.
    """
    return a + len(b)


from typing import Tuple, Union


def return_with_multiline_type_description_tuple_sig_type(
    a: int, b: str
) -> Tuple[int, str]:
    """Function with type description across multiple lines and Tuple type given in signature.

    Returns
    -------
    This integer is returned
    and represents the sum of a
    and length of b.
    """
    return a + len(b)


def return_with_multiline_type_description_union_sig_type(
    a: int, b: str
) -> Union[int, str]:
    """Function with type description across multiple lines and Union type given in signature.

    Returns
    -------
    This integer is returned
    and represents the sum of a
    and length of b.
    """
    return a + len(b)


def return_with_multiline_type_description_union2_sig_type(a: int, b: str) -> int | str:
    """Function with type description across multiple lines and Union2 type given in signature.

    Returns
    -------
    This integer is returned
    and represents the sum of a
    and length of b.
    """
    return a + len(b)


def return_with_multiline_type_description_tuple_field_type(a: int, b: str):
    """Function with type description across multiple lines and Tuple type given in Returns field.

    Returns
    -------
    Tuple[int, str]
        This integer is returned
        and represents the sum of a
        and length of b.
    """
    return a + len(b)


def return_with_multiline_type_description_union_field_type(a: int, b: str):
    """Function with type description across multiple lines and Union type given in Returns field.

    Returns
    -------
    Union[int, str]
        This integer is returned
        and represents the sum of a
        and length of b.
    """
    return a + len(b)


def return_with_multiline_type_description_union2_field_type(a: int, b: str):
    """Function with type description across multiple lines and Union2 type given in Returns field.

    Returns
    -------
    int | str
        This integer is returned
        and represents the sum of a
        and length of b.
    """
    return a + len(b)


def return_with_multiline_type_description_conflicting_type_01(a: int, b: str) -> int:
    """Function with type description across multiple lines and conflicting types.

    Returns
    -------
    int | str
        This integer is returned
        and represents the sum of a
        and length of b.
    """
    return a + len(b)


def return_with_multiline_type_description_conflicting_type_02(a: int, b: str) -> int:
    """Function with type description across multiple lines and conflicting types.

    Returns
    -------
    int | str
    """
    return a + len(b)


def return_with_multiline_type_description_conflicting_type_03(
    a: int, b: str
) -> int | str:
    """Function with type description across multiple lines and conflicting types.

    Returns
    -------
    int
        This integer is returned
        and represents the sum of a
        and length of b.
    """
    return a + len(b)


def return_with_formatted_return_01(
    a: int, b: str
) -> int | str:
    """Function with special formatting in Returns field.

    Returns
    -------
    int
        This integer is **returned**
        and represents the sum of `a`
        and length of '''b'''.
    """
    return a + len(b)

def return_with_formatted_return_02(
    a: int, b: str
) -> int | str:
    """Function with special formatting in Returns field.

    Returns
    -------
    This integer is **returned**
    and represents the sum of `a`
    and length of '''b'''.
    """
    return a + len(b)