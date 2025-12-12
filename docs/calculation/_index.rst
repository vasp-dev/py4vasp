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

..
    {% for group, members in calculation.GROUPS.items() %}
        {% for member in members %}
            ~py4vasp._calculation.{{ group }}_{{ member }}.
            {%- for part in group.split("_") -%}
                {{ part.capitalize() }}
            {%- endfor -%}
            {%- for part in member.split("_") -%}
                {{ part.capitalize() }}
            {%- endfor -%}
        {% endfor %}
    {% endfor %}

