from .models import BaseNeuralNetworkPotential
from loguru import logger as log
from openff.units import unit
import torch
from typing import Dict
from torch import nn

class PhysNetRepresentation(nn.Module):

    def __init__(
        self,
        cutoff: unit = 5 * unit.angstrom,
        number_of_gaussians: int = 16,
        device:torch.device = torch.device('cpu')
    ):
        super().__init__()

        # cutoff
        from modelforge.potential import CosineCutoff

        self.cutoff_module = CosineCutoff(cutoff, device)

        # radial symmetry function
        from .utils import RadialSymmetryFunction

        self.radial_symmetry_function_module = RadialSymmetryFunction(
            number_of_gaussians=number_of_gaussians,
            radial_cutoff=cutoff,
            ani_style=False,
            dtype=torch.float32,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]):
        """
        Generates the representation for the Physnet potential.

        Parameters
        ----------
        inputs (Dict[str, torch.Tensor]): A dictionary containing the input tensors.
            - "d_ij":  torch.Tensor, shape: (n_pairs, 1, distance)
            Pairwise distances between atoms.

        Returns:
        ----------
        f_ij: torch.Tensor:
            radial basis function expension, shape (n_pairs, 1, n_gaussians)
        """

        rbf = self.radial_symmetry_function_module(inputs["d_ij"])
        cutoff = self.cutoff_module(inputs["d_ij"])
        f_ij = torch.mul(rbf, cutoff)
        return f_ij


class PhysNetInteraction(nn.Module):

    def __init__():
        pass


    def forward(self, inputs: Dict[str, torch.Tensor])
    
        """
        PhysNetInteraction module. Takes as input the embedded nuclear charges 
        and the expended radial basis functions (already with cutoff added).

        Parameters:
        ----------
        inputs (Dict[str, torch.Tensor]): A dictionary containing the input tensors.
            - "f_ij": torch.Tensor, shape: (n_pairs, 1, n_gaussians)
                radial basis function expension, shape (n_pairs, 1, n_gaussians)
            - "atomic_embedding": torch.Tensor, shape: (nr_of_atoms_in_systems, 1, nr_atom_basis)
                Embeddings of atomic numbers. Shape: (n_atoms, embedding_dim).
            - "pairlist: torch.Tensor, shape: (2, nr_of_pairs)
                Pairlist of the systems.

        Returns:
            _type_: _description_
        """
        from .utils import ShiftedSoftplus
        softplus = ShiftedSoftplus()
        
        atom_embedding = inputs["atomic_embedding"]
        # Start with embedded features
        x = 

class PhysNetResidual(nn.Module):
    def __init__():
        super().__init__()

    def forward():
        pass

class PhysNetOutput(nn.Module):
    def __init__():
        super().__init__()
        
    def forward():
        pass

class PhysNetModule(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # this class combines the PhysNetInteraction, PhysNetResidual and 
        # PhysNetOutput class
        
        self.interaction = PhysNetInteraction()
        self.residula = PhysNetResidual()
        self.output = PhysNetOutput()
        
    def forward():
        pass

class PhysNet(BaseNeuralNetworkPotential):
    def __init__(
        self,
        max_Z: int = 100,
        embedding_dimensions: int = 64,
        cutoff: unit.Quantity = 5 * unit.angstrom,
        nr_of_modules : int = 2
    ) -> None:
        """
        Initialize the PhysNET class.
        Physnet is a neural network potential first described here: https://arxiv.org/pdf/1902.08408.pdf
        The potential takes as input the nuclear charges (Z_i) and pairwise distances (r_i)
        and outputs the atomic energies E_i and partial charges q_i.
        The architecture has the following components:
        - embedding layer: transforming nuclear charges into a vector representation
        - radial basis function layer: transforming the pairwise distances into a vector representation
        - module layer (see Fig 2 for graphical representation):
            embedding and radial basis output are provided as input,
            consists of (1) interaction, (2) residula and (3) output module
        Parameters
        ----------
        max_Z : nn.int
        embedding_dimensions : int
        """

        log.debug("Initializing PhysNET model.")

        self.only_unique_pairs = False  # NOTE: for pairlist
        super().__init__(cutoff=cutoff)
        self.nr_atom_basis = embedding_dimensions

        # embedding
        from modelforge.potential.utils import Embedding

        self.embedding_module = Embedding(max_Z, embedding_dimensions)

        # cutoff
        from modelforge.potential import CosineCutoff
        self.cutoff_module = CosineCutoff(cutoff, self.device)

        # initialize the energy readout
        from .utils import EnergyReadout

        self.readout_module = EnergyReadout(embedding_dimensions)

        self.physnet_representation_module = PhysNetRepresentation(
            radial_cutoff=cutoff, number_of_gaussians=16
        )

        # initialize the PhysNetModule building blocks
        from torch.nn import ModuleList
        self.physnet_module = PhysNetModule([PhysNetModule() for module_idx in range(nr_of_modules)])

    def _readout(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Compute the energy for each system
        return self.readout_module(inputs)

    def _model_specific_input_preparation(self, inputs: Dict[str, torch.Tensor]):
        # Perform atomic embedding
        inputs["atomic_embedding"] = self.embedding_module(inputs["atomic_numbers"])
        return inputs

    def _forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate the energy for a given input batch.
        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
        - atomic_embedding : torch.Tensor
            Atomic numbers embedding; shape (nr_of_atoms_in_systems, 1, nr_atom_basis).
        - pairlist:  shape (2, n_pairs)
        - d_ij:  shape (n_pairs, 1)
        - atomic_embedding:  shape (nr_of_atoms_in_systems, nr_atom_basis)
        Returns
        -------
        torch.Tensor
            Calculated energies; shape (nr_systems,).
        """
        
        # calculate rbf
        representation = self.physnet_representation_module(inputs["d_ij"])
        
        
        output = self.physnet_module(inputs, representation)

        self._readout(output)