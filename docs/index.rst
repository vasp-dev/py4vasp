py4vasp
=======

*py4vasp* is a python interface to extract data from Vasp calculations. It is
intended mainly to get a quick look at the data and provide functionality to
export it into common formats that can be used by other more sophisticated
postprocessing tools. A second domain of application is for people that want to
write python scripts based on the data calculated by Vasp. This tool interfaces
directly with the new HDF5 file format and thereby avoids parsing issues
associated with the XML or OUTCAR files.

For these two groups of users, we provide a different level of access. The
simple routines used in the tutorials will read the data from the file directly
and then generate the requested plot. For script developers, we provide an
expert interface were the data is lazily loaded as needed with some greater
flexibility when the data file is opened and closed.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
