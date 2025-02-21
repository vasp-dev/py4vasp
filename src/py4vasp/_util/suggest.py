# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import difflib


def did_you_mean(selection, possibilities):
    """Returns 'Did you mean X? ' if any X in `possibilities` is close to the `selection`
    input, otherwise it returns an empty string.

    Note that the trailing empty space allows including it in error messages directly.
    It will convert the inputs to strings to parse as much as possible.

    Parameters
    ----------
    selection : str
        The current selection used by the user, which is supposed to match one of the
        possibilities. The input is converted to string.
    possibilities : Sequence[str]
        A list of possible values for the selection. We find the closest match of the
        given selection to any of the given values. All values are converted to string.

    Returns
    -------
    str
        The string contains the closest possible match if any was found.
    """
    selection = str(selection)
    possibilities = [str(possibility) for possibility in possibilities]
    best_choice = difflib.get_close_matches(selection, possibilities, n=1)
    if best_choice:
        return f'Did you mean "{best_choice[0]}"? '
    else:
        return ""
