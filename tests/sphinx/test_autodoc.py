# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import sys

import pytest

application = pytest.importorskip("sphinx.application")


@pytest.fixture(scope="module")
def sphinx_app(tmp_path_factory, not_core):
    if sys.version_info <= (3, 11):
        pytest.skip("Sphinx autodoc tests require Python 3.12 or higher.")
    tmp_path = tmp_path_factory.mktemp("sphinx")
    srcdir = "tests/sphinx/autodoc"
    confdir = "tests/sphinx/autodoc"
    outdir = tmp_path / "_build"
    doctreedir = tmp_path / "_doctree"
    app = application.Sphinx(
        srcdir=srcdir,
        confdir=confdir,
        outdir=outdir,
        doctreedir=doctreedir,
        buildername="hugo",
        status=None,
        warning=None,
        freshenv=True,
    )
    app.build(force_all=True)
    return app


def read_file_content(outdir, source_file):
    output_file = outdir / "hugo" / source_file
    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert content.startswith("+++")
    return content


def test_autodoc_class(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "example_class.md")
    expected_content = """\
{{< class name="ExampleClass" module="example_class" breadcrumbs="" >}}
{{< signature >}}
(*value*)
{{< /signature >}}

{{< docstring >}}
This is an example class for testing Sphinx autodoc.

It demonstrates how to document a class with attributes and methods.

{{< /docstring >}}

{{< method name="get_value" class="ExampleClass" module="example_class" breadcrumbs="" >}}
{{< signature >}}
()
{{< /signature >}}

{{< docstring >}}
Return the stored value.

#### Returns
int
: <!---->
    The integer value stored in the instance.


{{< /docstring >}}


{{< /method >}}



{{< /class >}}"""
    assert expected_content in content


def test_autodoc_inner_class(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "inner_class.md")
    expected_content = """\
{{< class name="InnerClass" module="inner_class" breadcrumbs="inner" >}}
{{< signature >}}
(*data*)
{{< /signature >}}

{{< docstring >}}
An inner class for testing nested class documentation.

{{< /docstring >}}

{{< property name="get_data" class="InnerClass" module="inner_class" breadcrumbs="inner" >}}
{{< signature >}}: `str`
{{< /signature >}}

{{< docstring >}}
Retrieve the stored data.

#### Returns
str
: <!---->
    The data stored in the inner class.


{{< /docstring >}}


{{< /property >}}



{{< /class >}}"""
    assert expected_content in content


def test_autodoc_module(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "example_module.md")
    class_content = """\
{{< class name="ExampleModule" module="example_module" breadcrumbs="" >}}
{{< signature >}}
()
{{< /signature >}}

{{< docstring >}}
A simple example module to demonstrate Sphinx autodoc capabilities.

{{< /docstring >}}

{{< method name="example_method" class="ExampleModule" module="example_module" breadcrumbs="" >}}
{{< signature >}}
(
- *param1*,
- *param2*

)
{{< /signature >}}

{{< docstring >}}
An example function that adds two parameters.

#### Parameters
param1
int: <!---->
    The first parameter.

param2
int: <!---->
    The second parameter.


{{< /docstring >}}{{< docstring >}}
#### Returns
int
: <!---->
    The sum of param1 and param2.


{{< /docstring >}}


{{< /method >}}



{{< /class >}}"""
    assert class_content in content
    dataclass_content = """\
{{< class name="ExampleDataClass" module="example_module" breadcrumbs="" >}}
{{< signature >}}
(
- *attribute1*: `int`,
- *attribute2*: `str`

)
{{< /signature >}}

{{< docstring >}}
An example data class for testing Sphinx autodoc.

{{< /docstring >}}

{{< attribute name="attribute1" class="ExampleDataClass" module="example_module" breadcrumbs="" >}}
{{< signature >}}: `int`
{{< /signature >}}

{{< docstring >}}
An integer attribute.

{{< /docstring >}}

{{< /attribute >}}



{{< attribute name="attribute2" class="ExampleDataClass" module="example_module" breadcrumbs="" >}}
{{< signature >}}: `str`
{{< /signature >}}

{{< docstring >}}
A string attribute.

{{< /docstring >}}

{{< /attribute >}}



{{< /class >}}"""
    assert dataclass_content in content
    function_content = """\
{{< function name="example_function" module="example_module" breadcrumbs="" >}}
{{< signature >}}
(*x*)
{{< /signature >}}

{{< docstring >}}
An example function that squares its input.

### Parameters
x
int: <!---->
    An integer to be squared.


{{< /docstring >}}{{< docstring >}}
### Returns
int
: <!---->
    The square of the input integer.


{{< /docstring >}}


{{< /function >}}"""
    assert function_content in content


def test_autodoc_data(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "inner_data.md")
    expected_content = """\
{{< data name="inner_data" module="inner_data" breadcrumbs="inner" >}}

{{< docstring >}}
An inner class for testing nested class documentation.

{{< /docstring >}}

{{< /data >}}"""
    assert expected_content in content


def test_autosummary(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "index.md")
    expected_content = """\
[`ExampleClass`](example_class/ExampleClass)
: <!---->
    This is an example class for testing Sphinx autodoc.

[`ExampleModule`](example_module/ExampleModule)
: <!---->
    A simple example module to demonstrate Sphinx autodoc capabilities.

[`ExampleDataClass`](example_module/ExampleDataClass)
: <!---->
    An example data class for testing Sphinx autodoc.

[`InnerClass`](inner/inner_class/InnerClass)
: <!---->
    An inner class for testing nested class documentation.

[`inner_data`](inner/inner_data/inner_data)
: <!---->
    An inner class for testing nested class documentation."""
    assert expected_content in content


def test_multiple_autosummary(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "multiple_autosummary.md")
    expected_content = """\
{{< docstring >}}
This tests whether multiple autosummary directives are correctly processed.

[`ExampleClass`](../example_class/ExampleClass)
: <!---->
    This is an example class for testing Sphinx autodoc.

Text in between.

[`ExampleModule`](../example_module/ExampleModule)
: <!---->
    A simple example module to demonstrate Sphinx autodoc capabilities.

Text in between.

[`ExampleDataClass`](../example_module/ExampleDataClass)
: <!---->
    An example data class for testing Sphinx autodoc.

{{< /docstring >}}"""
    assert expected_content in content


@pytest.mark.skip(
    reason="Currently, parsing the three equivalent functions does not produce the same result."
)
def test_autodoc_equivalent_function(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "equivalent_function.md")
    expected_content = """\
{{< function name="METHOD_NAME" module="equivalent_function" breadcrumbs="" >}}
{{< signature >}}
() → `str`
{{< /signature >}}

{{< docstring >}}
docstring

### Returns
str
: <!---->
    bar


{{< /docstring >}}


{{< /function >}}"""
    assert expected_content.replace("METHOD_NAME", "foo1") in content
    assert expected_content.replace("METHOD_NAME", "foo2") in content
    assert expected_content.replace("METHOD_NAME", "foo3") in content
