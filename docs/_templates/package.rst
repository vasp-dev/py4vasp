{{ name | escape | underline }}

.. automodule:: {{ fullname }}

.. rubric:: Attributes

.. autosummary::
  :toctree:
  :template: member.rst

  {% for member in members %}
  {{ member }}
  {% endfor %}
