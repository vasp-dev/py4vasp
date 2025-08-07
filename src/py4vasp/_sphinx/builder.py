# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from sphinx.builders import Builder

from py4vasp._sphinx.translator import HugoTranslator


class HugoBuilder(Builder):
    """A Sphinx builder that generates Markdown files compatible with Hugo static site generator.

    This builder converts Sphinx documentation trees into Markdown format suitable for Hugo,
    enabling the generation of static websites from Sphinx documentation sources.

    Attributes:
        name (str): The builder name used in Sphinx configuration ('hugo').
        format (str): The output format ('markdown').
    """

    name = "hugo"
    format = "markdown"

    def get_outdated_docs(self):
        """Return which documents need to be rebuilt.

        Returns:
            str: Always returns "all documents" to rebuild everything.
        """
        return "all documents"

    def prepare_writing(self, docnames):
        """Prepare the output directory for writing documentation files.

        Creates the Hugo output directory structure before writing begins.

        Args:
            docnames: List of document names to be written.
        """
        self.hugo_dir = self.outdir / "hugo"
        self.hugo_dir.mkdir(parents=True, exist_ok=True)
        # return super().prepare_writing(docnames)

    def write_doc(self, docname, doctree):
        """Write a single document to a Markdown file.

        Converts the Sphinx document tree to Markdown format and writes it
        to a file in the Hugo output directory.

        Args:
            docname (str): The name of the document being written.
            doctree: The Sphinx document tree to convert.
        """
        filename = self.hugo_dir / f"{docname}.md"
        with open(filename, "w", encoding="utf-8") as outfile:
            markdown_content = self._doctree_to_markdown(doctree)
            outfile.write(markdown_content)

    def _doctree_to_markdown(self, doctree):
        """Convert a Sphinx document tree to Markdown format.

        Uses the HugoTranslator to walk through the document tree and
        generate Hugo-compatible Markdown content.

        Args:
            doctree: The Sphinx document tree to convert.

        Returns:
            str: The generated Markdown content as a string.
        """
        visitor = HugoTranslator(doctree)
        doctree.walkabout(visitor)
        return str(visitor)

    def get_target_uri(self, docname, typ=None):
        """Return the relative URI for a document.

        Args:
            docname (str): The name of the document.
            typ (str, optional): The type of the target, not used in this builder.

        Returns:
            str: The relative path to the Markdown file for the given document name.
        """
        # Return the relative path to the Markdown file for the given docname
        return f"hugo/{docname}.md"
