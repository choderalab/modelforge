Inference Mode
##########################

Inference mode is a mode allows us to use the trained model to make predictions. Given that a key usage of the inference mode will be molecule simulation, more efficient schemes for calculating interacting pairs are needed.

Neighborlists
------------------------------------------
Currently, there are two neighborlist strategies implemented within modelforge for inference, the brute force neighbolist and Verlet neighborlist (implemented within a single class :class:`~modelforge.potential.NeighborlistForInference`).    Both neighborlists support both periodic and not periodic orthogonal boxes.

The neighborlist strategy can be toggled during potential setup via the `inference_neighborlist_strategy` parameter passed to the :class:`~modelforge.potential.models.NeuralNetworkPotentialFactory`.  The default is the Verlet neighborlist ("verlet_nsq"); brute can be set via "brute_nsq".  This can also be set via set at run time in the potential via Potential.set_neighborlist_strategy(strategy, skin).

Brute force neighborlist
^^^^^^^^^^^^^^^^^^^^^^^^
The brute force neighborlist calculates the pairs within the interaction cutoff by considering all possible pairs each time called, via an order N^2 operation. Typically this approach should only be used for very system sizes, given the scaling; furthermore the N^2 approach used to generate this list utilizes a large amount of memory as the system size grows.



Verlet neighborlist
^^^^^^^^^^^^^^^^^^^^^^^^

The Verlet neighborlist operates under the assumption that under short time windows, the local environment around a given particle does not change significantly.  As such, information about this local environment can be reused between subsequent steps, eliminating the need for a costly build step.

To do this, the local environment of a given particle is identified and saved in a list (e.g., we can call this the verlet list), using the criteria pair distance < cutoff + skin.  The skin is a user modifiable distance that captures a region of space beyond the interaction cutoff.  In the current implementation, this verlet list is generated using the same order N^2 approach as the brute for scheme.  Again, because positions are correlated with time, we typically can avoid performing another order N^2 calculation for several timesteps.  Steps in between rebuilds scale as order N*M, where M is the average number of neighbors (which is typically much less than N).  In our implementation, the verlet list is automatically regenerated when any given particle moves more than skin/2 (since the last build), to ensure that interactions are not missed.

Larger values of skin result in longer time periods between rebuilds, but also typically increase the number of calculations that need to be perform at each timestep (as M will typically be larger). As such, this value can have a significant impact on performance of this calculation.

Note: Since this utilizes an N^2 computation within Torch, the memory footprint may be problematic as system size grows. A cell list based approach will be implemented in the future.


Load inference potential from training checkpoint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use the trained model for inference, the checkpoint file generated during training must be loaded. The checkpoint file contains the model's weights, optimizer state, and other training-related information. The `load_inference_model_from_checkpoint` function provides a convenient way to load the checkpoint file and generate an inference model. 

.. code-block:: python

    from modelforge.potential.models import load_inference_model_from_checkpoint

    inference_model = load_inference_model_from_checkpoint(checkpoint_file)



.. toctree::
   :maxdepth: 2
   :caption: Contents:

