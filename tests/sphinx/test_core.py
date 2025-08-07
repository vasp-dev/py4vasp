# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pytest

application = pytest.importorskip("sphinx.application")


@pytest.fixture(scope="module")
def sphinx_app(tmp_path_factory, not_core):
    tmp_path = tmp_path_factory.mktemp("sphinx")
    srcdir = "tests/sphinx/examples"
    confdir = "tests/sphinx/examples"
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


def test_register_hugo_builder(sphinx_app):
    assert "hugo" in sphinx_app.registry.builders


def read_file_content(outdir, source_file):
    output_file = outdir / "hugo" / source_file
    assert output_file.exists()
    content = output_file.read_text()
    assert content.startswith("+++")
    print(content)
    return content


def test_index_page(sphinx_app):
    """Test that the index page is created correctly."""
    content = read_file_content(sphinx_app.outdir, "index.md")
    assert 'title = "Main page"' in content
    assert "# Main page" in content
    assert (
        "This is the main page of the documentation. It serves as an index for the content available in this project."
        in content
    )


def test_convert_comment(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "comments.md")
    assert "This is visible content and should appear in the output." in content
    assert "This is a comment and should not appear in the output." not in content


def test_convert_paragraphs(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "paragraph.md")
    doctree = sphinx_app.env.get_doctree("paragraph")
    # markdown needs double line break for new paragraph
    assert "aliqua.\n\nUt" in content


def test_convert_headings(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "headings.md")
    assert 'title = "Chapter"' in content
    lines = content.splitlines()
    # find all headers
    headers = []
    headers.append(lines.index("# Chapter"))
    for section in range(1, 3):
        section_header = f"## Section {section}"
        headers.append(lines.index(section_header))
        for subsection in range(1, 3):
            subsection_header = f"### Subsection {section}.{subsection}"
            headers.append(lines.index(subsection_header))
            for subsubsection in range(1, 3):
                subsubsection_header = (
                    f"#### Subsubsection {section}.{subsection}.{subsubsection}"
                )
                headers.append(lines.index(subsubsection_header))
                for paragraph in range(1, 3):
                    paragraph_header = f"##### Paragraph {section}.{subsection}.{subsubsection}.{paragraph}"
                    headers.append(lines.index(paragraph_header))
    # check that headers are in order
    assert sorted(headers) == headers
    assert "***Header not in table of contents***\n" in content


def test_convert_inline_markup(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "inline_markup.md")
    expected_content = f"""\
+++
title = "Inline markup example"
+++
# Inline markup example
*this text is emphasized*, **this text is strong**, `this text is code`
"""
    assert expected_content in content


def test_convert_lists(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "list.md")
    unordered_list = """\
* this is
* a list
  * with a nested list
  * and some subitems
* and here the parent list continues"""
    assert unordered_list in content
    ordered_list = """\
1. This is a numbered list.
1. It has two items too.
1. This is a numbered list.
1. It has two items, the second
item uses two lines."""
    assert ordered_list in content
    definition_list = """\
term (up to a line of text)
: Definition of the term, which must be indented
and can even consist of multiple paragraphs

next term
: Description."""
    assert definition_list in content


def test_convert_reference(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "reference.md")
    # doctree = sphinx_app.env.get_doctree("reference")
    # print(doctree.pformat())
    ## CAREFUL: internal references are first shown as pending_xref nodes,
    ## then converted to inline nodes with class "xref std std-ref"
    ## therefore, the processing by the translator must be done in visit/depart_inline

    # External links
    assert "This is a link to the [VASP website](https://www.vasp.at)." in content
    assert "This is a different [external link](https://www.example.com)." in content

    # Internal reference (Markdown link to anchor)
    assert "[internal-label](#internal-label)" in content
    # Internal anchor
    assert '<a name="internal-label"></a>' in content
    # Target section text
    assert "This is the internal target section." in content

def test_convert_compound(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "compound.md")
    expected_content = """\
<div class="compound">

This is inside a compound block.

Another paragraph inside compound.


</div>
"""
    assert expected_content in content
    expected_content_nested = """\
<div class="compound">

This is inside a compound block.


<div class="compound">

This is a nested compound block.


</div>

</div>
"""
    assert expected_content_nested in content

def test_convert_custom_role(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "custom_role.md")
    assert "[LORBIT](https://vasp.at/wiki/index.php/LORBIT)" in content


def test_code_block(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "code_block.md")
    default_code_block = """\
~~~
This is a generic code block.

It can contain any text, including special characters like $ and %.
~~~"""
    assert default_code_block in content
    bash_code_block = """\
~~~bash
echo "Hello, World!"
~~~"""
    assert bash_code_block in content
    python_code_block = """\
~~~python
print("Hello, World!")
~~~"""
    assert python_code_block in content
    doctest_code_block = """\
~~~python
>>> print("Hello, World!")
Hello, World!
~~~"""
    assert doctest_code_block in content


def test_convert_admonition(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "admonition.md")
    assert "[!note]\nThis is a note admonition." in content
    assert "[!warning]\nThis is a warning admonition." in content
    assert "[!caution]\nThis is a caution admonition." in content
    assert "[!tip]\nThis is a tip admonition." in content
    assert "[!important]\nThis is an important admonition." in content


@pytest.mark.skip("Indentation handling needs to be fixed")
def test_convert_footnote(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "footnote.md")
    print(content)
    assert False

#@pytest.mark.skip("WiP")
def test_convert_example(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "example.md")
    expected_autodata_content = """\

<div class='data signature'>

<a id='example.Example'></a>

## *data* **Example** [¶](#example.Example)

</div>



<div class='desc-content'>

An example class for demonstration purposes.


</div>

"""
    assert expected_autodata_content in content

    expected_class_content = """\

<div class='class signature'>

<a id='example.Example'></a>

## *class* **Example** [¶](#example.Example)(*value*: `float`)

</div>



<div class='desc-content'>

Bases: `object`

An example class for demonstration purposes.

"""
    assert expected_class_content in content

    expected_private_method_content = """\

<div class='method signature'>

<a id='example.Example.__init__'></a>

### **__init__** [¶](#example.Example.__init__)(*value*: `float`)

</div>



<div class='desc-content'>

Initialize the Example class with a value.

* some list entry
* some other list entry



#### **Parameters:**

- *value*: `float`
: The value to be stored in the instance.


</div>

"""
    assert expected_private_method_content in content

    expected_public_method_content = """\

<div class='method signature'>

<a id='example.Example.combined_returns'></a>

### **combined_returns** [¶](#example.Example.combined_returns)
(
- *some_value*: `float`,
- *some_string*: `str | None`, optional [default: '']

) → `tuple[float, str | None]`

</div>



<div class='desc-content'>

Combine a float and a string in a tuple.



#### **Parameters:**

- *some_value*: `float`
: A value to be included in the tuple.
- *some_string*: `str`, optional [default: '']
: A string to be included in the tuple.

#### **Returns:**

- `tuple[float, str | None]`
: A tuple containing the float and a string representation.



</div>

"""
    assert expected_public_method_content in content