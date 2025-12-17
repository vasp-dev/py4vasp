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
