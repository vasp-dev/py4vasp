# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)


def tag_block(tag, rows, group=None):
    """Format an INCAR tag whose value is a list of numbers, one row per line.

    The lines are joined by a backslash, which VASP reads as a line continuation. If
    a group size is given, a wider gap separates every group of that many numbers
    within a line, e.g. the rows of a 3x3 tensor written on a single line.
    """
    prefix = f"{tag} = "
    lines = (_format_row(row, group) for row in rows)
    separator = " \\\n" + len(prefix) * " "
    return prefix + separator.join(lines) + "\n"


def _format_row(row, group):
    numbers = [f"{x:10.6f}" for x in row]
    group = group or len(numbers)
    groups = (" ".join(numbers[i : i + group]) for i in range(0, len(numbers), group))
    return "   ".join(groups)
