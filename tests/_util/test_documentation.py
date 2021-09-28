import py4vasp._util.documentation as _documentation
import inspect


def test_add_documentation():
    doc_string = "doc string"

    @_documentation.add(doc_string)
    def func():
        pass

    assert inspect.getdoc(func) == doc_string
