# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pytest

application = pytest.importorskip("sphinx.application")


@pytest.fixture(scope="module")
def sphinx_app(tmp_path_factory, not_core):
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
