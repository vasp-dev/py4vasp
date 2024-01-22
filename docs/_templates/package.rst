{{ name | escape | underline }}

.. automodule:: {{ fullname }}

.. autosummary::
  :toctree:
  :template: module.rst

  {% for module in modules %}
  {{ module.split(".") | last }}
  {% endfor %}
