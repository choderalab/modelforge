For Developers
===============

.. note::
    
        This section is intended for developers who want to extend the functionality of the `modelforge` package
        or who want to develop their own machine learned potentials. This section explains design decicions 
        and the structure of neural network potentials.



How to deal with units
---------------------------------

All public APIs require explicit units for values that are not dimensionless.
The units are specified using the `openff.units` package, e.g.:

.. code-block:: python
    
        from openff.units import unit
    
        # A length of 1.0 angstrom
        length = 1.0 * unit.angstrom
    


Internally, when units are removed, we use the openmm units system 
`here <http://docs.openmm.org/latest/userguide/theory/01_introduction.html#units/>`_.

To demonstrate this we can look at the `CosineCutoff` class, 
which is used to add a smooth cutoff to a distance tensor. 
The cutoff is specified in units of the length tensor, e.g.:
`unit.nanometer`. The `d_ij` tensor is passed in nanometer (but without 
explicit attached units)..

.. autoclass:: modelforge.potential.utils.CosineCutoff
    :members:

Base structure of machine learned potentials
-------------------------------------------------

The base structure of machine learned potentials is as shown in the figure below.


The `NetworkWrapper` class is the main class that is used to store the potential.
The `CoreNetwork` class is used to store the neural network that is used to predict the potential.


.. image:: image/overview_network.png
  :width: 400
  :align: center
  :alt: Alternative text


.. autoclass:: modelforge.potential.models.NetworkWrapper
    :members:


.. autoclass:: modelforge.potential.models.CoreNetwork
    :members:

.. autoclass:: modelforge.potential.models.InputPreparation
    :members:
