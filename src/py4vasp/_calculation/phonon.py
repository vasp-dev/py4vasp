# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
selection_doc = """\
selection : str
    A string specifying the projection of the phonon modes onto atoms and directions.
    Please specify selections using one of the following:

    -   To specify the **atom**, you can either use its element name (Si, Al, ...)
        or its index as given in the input file (1, 2, ...). For the latter
        option it is also possible to specify ranges (e.g. 1:4).
    -   To select a particular **direction** specify the Cartesian direction (x, y, z).

    You separate multiple selections by commas or whitespace and can nest them using
    parenthesis, e.g. `Sr(x)` or `z(1, 2)`. The order of the selections does not matter,
    but it is case sensitive to distinguish y (Cartesian direction) from Y (yttrium).
    You can also add or subtract different selections e.g. `Sr - Ti`.

    If you are unsure what selections exist, please use the `selections` routine which
    will return all possibilities.
"""
