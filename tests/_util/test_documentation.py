# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import py4vasp._util.documentation as _documentation
import inspect


def test_add_documentation():
    doc_string = "doc string"

    @_documentation.add(doc_string)
    def func():
        pass

    assert inspect.getdoc(func) == doc_string


def test_format_documentation():
    inner_text = """\
First line
    Second line
"""

    @_documentation.format(inner_text=inner_text)
    def func():
        """Multiple line string

        {inner_text}

        continued afterwards"""

    expected = """\
Multiple line string

First line
    Second line

continued afterwards"""
    assert inspect.getdoc(func) == expected
