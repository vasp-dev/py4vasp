Calculation
===========

.. autoclass:: py4vasp.Calculation
   :members: from_path, from_file, path

Available quantities
--------------------

.. jinja::
    .. autosummary::
        :nosignatures:
    {% for autosummary in calculation.AUTOSUMMARIES %}
        {{ autosummary[1] }}
    {% endfor %}