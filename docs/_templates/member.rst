{{ name | escape | underline }}

.. container:: quantity

   .. currentmodule:: py4vasp.calculation
   .. data:: {{ name }}

   .. currentmodule:: py4vasp.data
   .. autoclass:: {%
      if name == "CONTCAR" -%}
         CONTCAR
      {%- else -%}
      {%- for part in name.split("_") -%}
         {{ part.capitalize() }}
      {%- endfor -%}
      {%- endif %}
      :members:
      :inherited-members:
      :exclude-members: from_data, from_file, from_path, path
