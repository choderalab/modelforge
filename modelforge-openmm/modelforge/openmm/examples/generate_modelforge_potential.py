"""
generate_modelforge_potential.py

This script generates a modelforge potential from a dataset and saves it as a torchscript model.
This is used to generate a potential for testing interactions with OpenMM.

"""

# use the helper function to load up the toml files into pydantic models
# we will only use the "potential" parameters, so we can provide any valid dataset name
from modelforge.utils.misc import load_configs_into_pydantic_models

config = load_configs_into_pydantic_models("ani2x", "phalkethoh")

# generate the ani2x potential using the NeuralNetworkPotentialFactory
# note this is not a trained potential

from modelforge.potential.potential import NeuralNetworkPotentialFactory

potential = NeuralNetworkPotentialFactory.generate_potential(
    potential_parameter=config["potential"],
    potential_seed=42,
    jit=False,
)

import torch

# convert the potential to a torchscript model and save it
jit_potential = torch.jit.script(potential)
jit_potential.save("data/ani2x_test.pt")

#
#
# from modelforge.dataset.dataset import single_batch
#
# # load in a single data
# batch = single_batch(batch_size=1, dataset_name="qm9", local_cache_dir="./")
#
