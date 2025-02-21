# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import difflib


def did_you_mean(actual, possibilities):
    """Returns 'Did you mean X? ' if any X in `possibilities` is close to the `actual`
    input, otherwise it returns an empty string.

    Note that the trailing empty space allows including it in error messages directly.
    """
    best_choice = difflib.get_close_matches(actual, possibilities, n=1)
    if best_choice:
        return f'Did you mean "{best_choice[0]}"? '
    else:
        return ""
