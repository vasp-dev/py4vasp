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

def test_index_page(sphinx_app):
    """Test that the index page is created correctly."""
    output_file = sphinx_app.outdir / "hugo/index.md"
    assert output_file.exists()
    content_text = output_file.read_text()
    print(content_text)
    assert content_text.startswith("+++")
    assert 'title = "Main page"' in content_text
    assert '# Main page' in content_text
    assert 'This is the main page of the documentation. It serves as an index for the content available in this project.' in content_text

def test_convert_example(sphinx_app):
    output_file = sphinx_app.outdir / "hugo/example.md"
    assert output_file.exists()
    content = output_file.read_text()
    assert content.startswith("+++")
    print(content)

def test_convert_comment(sphinx_app):
    output_file = sphinx_app.outdir / "hugo/comments.md"
    assert output_file.exists()
    content = output_file.read_text()
    print(content)
    assert content.startswith("+++")
    assert 'title = "Comment"' in content
    assert 'This is visible content' in content
    assert not('This is a comment.' in content)


def read_file_content(outdir, source_file):
    output_file = outdir / "hugo" / source_file
    assert output_file.exists()
    content = output_file.read_text()
    assert content.startswith("+++")
    return content


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


def test_convert_inline_markup(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "inline_markup.md")
    expected_content = f"""\
# Inline markup example

*this text is emphasized*, **this text is strong**, `this text is code`"""
    assert expected_content in content


def test_convert_lists(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "list.md")
    print(content)
    #     expected_content = f"""\
    # # Lists example

    # * Item 1
    # * Item 2
    #   * Subitem 2.1
    #   * Subitem 2.2
    # * Item 3
    # """
    # assert expected_content in content
    assert False
