calculation
===========

.. autodata:: py4vasp.calculation
   :annotation:

Available quantities
--------------------

.. jinja::
    .. autosummary::
        :nosignatures:
    {% for quantity in calculation.QUANTITIES %}
        {% if not quantity.startswith("_") -%}
            ~py4vasp.Calculation.{{ quantity }}
        {%- endif -%}
    {% endfor %}

Available groups
----------------
.. jinja::
    {% for group, members in calculation.GROUPS.items() %}
    .. rubric:: {{ group }}

    .. autosummary::
        :nosignatures:
        {% for member in members %}
            ~py4vasp.Calculation.{{ group }}.{{ member }}
        {% endfor %}

    {% endfor %}

