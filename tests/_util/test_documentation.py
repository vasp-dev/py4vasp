from py4vasp._util.documentation import _add_documentation
import inspect


def test_add_documentation():
    doc_string = "doc string"

    @_add_documentation(doc_string)
    def func():
        pass

    assert inspect.getdoc(func) == doc_string
