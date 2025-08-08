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


def read_file_content(outdir, source_file):
    output_file = outdir / "hugo" / source_file
    assert output_file.exists()
    content = output_file.read_text()
    assert content.startswith("+++")
    return content


def test_convert_example_autodata(sphinx_app):
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


def test_convert_example_autoclass_signature(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "example.md")
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


def test_convert_example_init_signature(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "example.md")
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


*value*: `float`
: <!---->
    The value to be stored in the instance.
    
</div>



</div>

"""
    assert expected_private_method_content in content


def test_convert_example_combined_returns_signature(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "example.md")
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


*some_value*: `float`
: <!---->
    A value to be included in the tuple.

*some_string*: [optional] `str` [default: '']
: <!---->
    A string to be included in the tuple.
    
</div>


#### **Returns:**

<div class='desc-content'>

`tuple[float, str | None]`
: <!---->
    A tuple containing the float and a string representation.
    


</div>

"""
    assert expected_public_method_content in content


def test_convert_example_returns_type_without_desc_returns_signature(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "example_return_types.md")
    expected_method_content = """\

<div class='function signature'>

<a id='example.returns_type_without_desc_returns'></a>

## *function* **returns_type_without_desc_returns** [¶](#example.returns_type_without_desc_returns)
(
- *value1*: `float`,
- *value2*: `float | str`

) → `float | str`

</div>



<div class='desc-content'>

Return value 2.



### **Parameters:**

<div class='desc-content'>


*value1*: `float`
: <!---->
    The first value.

*value2*: `float | str`
: <!---->
    The second value, which can be a float or a string.
    
</div>


### **Returns:**

<div class='desc-content'>

`float | str`
: <!---->
    The second value.
    
"""
    assert expected_method_content in content


def test_convert_example_returns_type_without_returns_field_signature(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "example_return_types.md")
    expected_method_content = """\

<div class='function signature'>

<a id='example.returns_type_without_returns_field'></a>

## *function* **returns_type_without_returns_field** [¶](#example.returns_type_without_returns_field)
(
- *value1*: `float`,
- *value2*: `float | str`

) → `float | str`

</div>



<div class='desc-content'>

Return value 2.



### **Parameters:**

<div class='desc-content'>


*value1*: `float`
: <!---->
    The first value.

*value2*: `float | str`
: <!---->
    The second value, which can be a float or a string.
    
</div>


### **Returns:**

<div class='desc-content'>

`float | str`

</div>


</div>

"""
    assert expected_method_content in content


def test_convert_example_returns_type_without_returns_field_type_signature(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "example_return_types.md")
    print(content)
    expected_method_content = """\

<div class='function signature'>

<a id='example.returns_type_without_returns_field_type'></a>

## *function* **returns_type_without_returns_field_type** [¶](#example.returns_type_without_returns_field_type)
(
- *value1*: `float`,
- *value2*: `float | str`

) → `float | str`

</div>



<div class='desc-content'>

Return value 2.



### **Parameters:**

<div class='desc-content'>


*value1*: `float`
: <!---->
    The first value.

*value2*: `float | str`
: <!---->
    The second value, which can be a float or a string.
    
</div>


### **Returns:**

<div class='desc-content'>

`float | str`
: <!---->
    The second value.
    With another line!
    
    


</div>



</div>

"""
    assert expected_method_content in content


def test_convert_example_returns_type_without_returns_field_desc_signature(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "example_return_types.md")
    print(content)
    expected_method_content = """\
<div class='function signature'>

<a id='example.returns_type_without_returns_field_desc'></a>

## *function* **returns_type_without_returns_field_desc** [¶](#example.returns_type_without_returns_field_desc)
(
- *value1*: `float`,
- *value2*: `float | str`

) → `float | str`

</div>



<div class='desc-content'>

Return value 2.



### **Parameters:**

<div class='desc-content'>


*value1*: `float`
: <!---->
    The first value.

*value2*: `float | str`
: <!---->
    The second value, which can be a float or a string.
    
</div>


### **Returns:**

<div class='desc-content'>

`float | str`

</div>


</div>
"""
    assert expected_method_content in content


# @pytest.mark.skip
def test_convert_example_dos(sphinx_app):
    content = read_file_content(sphinx_app.outdir, "example_dos.md")
