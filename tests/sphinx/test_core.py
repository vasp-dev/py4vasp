# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pytest

from py4vasp._util import import_

application = import_.optional("sphinx.application")


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


def test_convert_headings(sphinx_app):
    output_file = sphinx_app.outdir / "hugo/headings.md"
    assert output_file.exists()
    content = output_file.read_text()
    assert content.startswith("+++")
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
