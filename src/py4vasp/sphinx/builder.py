# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from sphinx.builders import Builder

from py4vasp.sphinx.translator import HugoTranslator


class HugoBuilder(Builder):
    name = "hugo"
    format = "markdown"

    def get_outdated_docs(self):
        return "all documents"

    def prepare_writing(self, docnames):
        self.hugo_dir = self.outdir / "hugo"
        self.hugo_dir.mkdir(parents=True, exist_ok=True)
        # return super().prepare_writing(docnames)

    def write_doc(self, docname, doctree):
        filename = self.hugo_dir / f"{docname}.md"
        with open(filename, "w", encoding="utf-8") as outfile:
            markdown_content = self._doctree_to_markdown(doctree)
            outfile.write(markdown_content)

    def _doctree_to_markdown(self, doctree):
        visitor = HugoTranslator(doctree)
        doctree.walkabout(visitor)
        return "".join(visitor.content)
