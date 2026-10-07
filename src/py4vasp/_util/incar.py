# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)


def tag_block(tag, rows):
    """Format an INCAR tag whose value is a list of numbers, one row per line.

    The lines are joined by a backslash, which VASP reads as a line continuation.
    """
    prefix = f"{tag} = "
    lines = (" ".join(f"{x:10.6f}" for x in row) for row in rows)
    separator = " \\\n" + len(prefix) * " "
    return prefix + separator.join(lines) + "\n"
