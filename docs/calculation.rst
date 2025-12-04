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
            ~py4vasp._calculation.{{ quantity }}.
            {%- for part in quantity.split("_") -%}
                {{ part.capitalize() }}
            {%- endfor -%}
        {%- endif -%}
    {% endfor %}

.. autoclass:: py4vasp.Calculation
   :members: from_path, from_file, path

.. .. autosummary::
    :recursive:
    py4vasp.Calculation
