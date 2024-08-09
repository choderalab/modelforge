For Developers
===============

.. note::
    
        This section is intended for developers who want to extend the functionality of the `modelforge` package or who want to develop their own machine learned potentials. This section explains design decisions
        and the structure of neural network potentials.



How to deal with units
---------------------------------

All public APIs require explicit units for values that are not dimensionless.
The units are specified using the `openff.units` package, e.g.:

.. code-block:: python
    
        from openff.units import unit
    
        # A length of 1.0 angstrom
        length = 1.0 * unit.angstrom
    

Units are also provided in the TOML files. For example, the following TOML file specifies a maximum interaction radius of 5.1 angstrom:

.. code-block:: toml

        maximum_interaction_radius = "5.1 angstrom"


Internally, when units are removed, we use the openmm units system 
`here <http://docs.openmm.org/latest/userguide/theory/01_introduction.html#units/>`_.


Base structure of machine learned potentials
-------------------------------------------------

The base structure of machine learned potentials is as shown in the figure
below. The :py:class:`~modelforge.potential.models.BaseNetwork` class
encapsulates the neighbor list calculation
:py:class:`~modelforge.potential.models.ComputeInteractingAtomPairs` (including
distances and distance vector for each atom pair) and  the neural network
potential :py:class:`~modelforge.potential.models.CoreNetwork`.
The :py:class:`~modelforge.potential.models.CoreNetwork` provides as output a variable number of scalars (e.g. per atom energies `E_i` and partial charges `q_i`). It also includes the feature representation of the atoms before it is passed through the readout layer (which is unique to each scalar property). 

The :py:class:`~modelforge.potential.model.Postprocessing` is used for reduction operations to obtain the molecular energy. There are other postprocessing operations that are performed there (calculation molecular self energy, etc).

.. image:: image/overview_network.png
  :width: 400
  :align: center
  :alt: Alternative text

The operations in the :py:class:`~modelforge.potential.models.CoreNetwork` can be seperated in a representation module, which encodes information about the environment and the atomic numbers, and an interaction module, which iterativley learns to update an atomic feature representation based on the local, pairiwse interactions between atoms.  

.. image:: image/overview_core_network.png
  :width: 400
  :align: center
  :alt: Alternative text

