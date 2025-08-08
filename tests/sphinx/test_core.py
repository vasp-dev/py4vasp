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
    # markdown needs double line break for new paragraph
    assert "aliqua.\n\nUt" in content


def test_convert_headings(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "headings.md")
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
    assert "###### ***Header not in table of contents***\n" in content


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

* and here the parent list continues

  with multiple paragraphs"""
    assert unordered_list in content
    ordered_list = """\
1. This is a numbered list.

1. It has two items too.

1. This is a numbered list.

1. It has two items, the second
  item uses two lines."""
    assert ordered_list in content
    # Note we added a # after the colon in the definition list so that Hugo renders it
    # correctly. Technically only the space is needed, but then opening the file manually
    # in a text editor may trim the trailing space.
    definition_list = """\
term (up to a line of text)
: <!---->
    Definition of the term, which must be indented

    and can even consist of multiple paragraphs

next term
: <!---->
    Description."""
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
    note_admonition = """\
{{< admonition type="info" >}}
This is a note admonition.
{{< /admonition >}}"""
    assert note_admonition in content
    warning_admonition = """\
{{< admonition type="warning" >}}
This is a warning admonition.
{{< /admonition >}}"""
    assert warning_admonition in content
    caution_admonition = """\
{{< admonition type="danger" >}}
This is a caution admonition.
{{< /admonition >}}"""
    assert caution_admonition in content
    tip_admonition = """\
{{< admonition type="success" >}}
This is a tip admonition.
{{< /admonition >}}"""
    assert tip_admonition in content
    important_admonition = """\
{{< admonition type="primary" >}}
This is an important admonition.
{{< /admonition >}}"""
    assert important_admonition in content


def test_convert_footnote(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "footnote.md")
    first_reference = "documentation.[^1]"
    assert first_reference in content
    second_reference = "reference.[^2]"
    assert second_reference in content
    third_reference = "footnote.[^3]"
    assert third_reference in content
    first_footnote = """\
[^1]:
    This is the first footnote.
    It can contain multiple lines of text and even
    some formatting like *bold* or **italic**."""
    assert first_footnote in content
    second_footnote = """\
[^2]:
    This is the second footnote.

    The second footnote has multiple paragraphs and a code block:

    ~~~python
    print("This is a code block in a footnote.")
    ~~~"""
    assert second_footnote in content
    third_footnote = """\
[^3]:
    This is the third footnote with a definition list.

    term
    : <!---->
        Definition of the term in the third footnote.

    next term
    : <!---->
        Description of the next term in the third footnote."""
    assert third_footnote in content


def test_complex_list(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "complex_list.md")
    # note that there is a space after the colon in the definition list
    # otherwise Hugo will not render it correctly
    list_with_codeblock = """\
title
: <!---->
    ~~~python
    print("This is a code block in a definition list.")
    ~~~

    text after the code block"""
    assert list_with_codeblock in content
    list_with_nested_list = """\
* first term
  : <!---->
      definition

  second term
  : <!---->
      definition with a list

    * item 1

    * item 2

* second list item

  term in paragraph
  : <!---->
      first paragraph

      second paragraph"""
    assert list_with_nested_list in content
    assert False


# def test_convert_p4v(sphinx_app):
#     content = read_file_content(sphinx_app.outdir, "p4v.md")
#     print(content)
#     assert False
#     # doctree = sphinx_app.env.get_doctree("reference")
#     # print(doctree.pformat())


@pytest.mark.skip
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

<div class='desc-content'>

- *value*: `float`
: The value to be stored in the instance.

</div>



</div>

"""
    assert expected_private_method_content in content

    expected_public_method_content = """\

<div class='method signature'>

<a id='example.Example.combined_returns'></a>

### **combined_returns** [¶](#example.Example.combined_returns)
(
- *some_value*: `float`,
- *some_string*: [optional] `str | None` [default: '']

) → `tuple[float, str | None]`

</div>



<div class='desc-content'>

Combine a float and a string in a tuple.



#### **Parameters:**

<div class='desc-content'>

- *some_value*: `float`
: A value to be included in the tuple.
- *some_string*: [optional] `str` [default: '']
: A string to be included in the tuple.

</div>


#### **Returns:**

<div class='desc-content'>

- `tuple[float, str | None]`
: A tuple containing the float and a string representation.

</div>



</div>

"""
    assert expected_public_method_content in content


@pytest.mark.skip
def test_convert_example_dos(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "example_dos.md")
    print(content)
