import torch
import torch.nn as nn
from typing import Dict, List, Union

from modelforge.potential.utils import DenseWithCustomDist
from modelforge.dataset import NNPInput


class AddPerMoleculeValue(nn.Module):
    """
    Module that adds a per-molecule value to a per-atom property tensor.
    The per-molecule value is expanded to match th elength of the per-atom property tensor.

    Parameters
    ----------
    key : str
        The key to access the per-molecule value from the input data.

    Attributes
    ----------
    key : str
        The key to access the per-molecule value from the input data.
    """

    def __init__(self, key: str):
        super().__init__()
        self.key = key

    def forward(
        self, per_atom_property_tensor: torch.Tensor, data: NNPInput
    ) -> torch.Tensor:
        """
        Forward pass of the module.

        Parameters
        ----------
        per_atom_property_tensor : torch.Tensor
            The per-atom property tensor.
        data : NNPInput
            The input data containing the per-molecule value.

        Returns
        -------
        torch.Tensor
            The updated per-atom property tensor with the per-molecule value appended.
        """
        values_to_append = getattr(data, self.key)
        _, counts = torch.unique(data.atomic_subsystem_indices, return_counts=True)
        expanded_values = torch.repeat_interleave(values_to_append, counts).unsqueeze(1)
        return torch.cat((per_atom_property_tensor, expanded_values), dim=1)


class AddPerAtomValue(nn.Module):
    """
    Module that adds a per-atom value to a tensor.

    Parameters
    ----------
    key : str
        The key to access the per-atom value from the input data.

    Attributes
    ----------
    key : str
        The key to access the per-atom value from the input data.
    """

    def __init__(self, key: str):
        super().__init__()
        self.key = key

    def forward(
        self, per_atom_property_tensor: torch.Tensor, data: NNPInput
    ) -> torch.Tensor:
        """
        Forward pass of the module.

        Parameters
        ----------
        per_atom_property_tensor : torch.Tensor
            The input tensor representing per-atom properties.
        data : NNPInput
            The input data object containing additional information.

        Returns
        -------
        torch.Tensor
            The tensor with the per-atom value appended.
        """
        values_to_append = getattr(data, self.key)
        return torch.cat((per_atom_property_tensor, values_to_append), dim=1)


class FeaturizeInput(nn.Module):

    _SUPPORTED_FEATURIZATION_TYPES = [
        "atomic_number",
        "per_system_total_charge",
        "spin_state",
    ]

    def __init__(self, featurization_config: Dict[str, Dict[str, int]]) -> None:
        """
        Initialize the FeaturizeInput class.

        For per-atom non-categorical properties and per-molecule properties
        (both categorical and non-categorical), we append the embedded nuclear
        charges and mix them using a linear layer.

        For per-atom categorical properties, we define an additional embedding
        and add the embedding to the nuclear charge embedding.

        Parameters
        ----------
        featurization_config : dict
            A dictionary containing the featurization configuration. It should
            have the following keys:
            - "properties_to_featurize" : list
                A list of properties to featurize.
            - "maximum_atomic_number" : int
                The maximum atomic number.
            - "number_of_per_atom_features" : int
                The number of per-atom features.

        Returns
        -------
        None
        """
        super().__init__()

        # expend embedding vector
        self.append_to_embedding_tensor = nn.ModuleList()
        self.registered_appended_properties: List[str] = []
        # what other categorial properties are embedded
        self.embeddings = nn.ModuleList()
        self.registered_embedding_operations: List[str] = []

        self.increase_dim_of_embedded_tensor: int = 0
        base_embedding_dim = int(
            featurization_config["atomic_number"]["number_of_per_atom_features"]
        )
        properties_to_featurize = featurization_config["properties_to_featurize"]
        # iterate through the supported featurization types and check if one of
        # these is requested
        for featurization in properties_to_featurize:

            # embed atomic number
            if (
                featurization == "atomic_number"
                and featurization in self._SUPPORTED_FEATURIZATION_TYPES
            ):
                self.atomic_number_embedding = torch.nn.Embedding(
                    int(featurization_config[featurization]["maximum_atomic_number"]),
                    int(
                        featurization_config[featurization][
                            "number_of_per_atom_features"
                        ]
                    ),
                )
                self.registered_embedding_operations.append("atomic_number")

            # add total charge to embedding vector
            elif (
                featurization == "per_system_total_charge"
                and featurization in self._SUPPORTED_FEATURIZATION_TYPES
            ):
                # transform output o f embedding with shape (nr_atoms,
                # nr_features) to (nr_atoms, nr_features + 1). The added
                # features is the total charge (which will be transformed to a
                # per-atom property)
                self.append_to_embedding_tensor.append(
                    AddPerMoleculeValue("per_system_total_charge")
                )
                self.increase_dim_of_embedded_tensor += 1
                self.registered_appended_properties.append("per_system_total_charge")

            # add partial charge to embedding vector
            elif (
                featurization == "per_atom_partial_charge"
                and featurization in self._SUPPORTED_FEATURIZATION_TYPES
            ):  # transform output of embedding with shape (nr_atoms, nr_features) to (nr_atoms, nr_features + 1).
                # #The added features is the total charge (which will be
                # transformed to a per-atom property)
                self.append_to_embedding_tensor.append(
                    AddPerAtomValue("partial_charge")
                )
                self.increase_dim_of_embedded_tensor += 1
                self.append_to_embedding_tensor("partial_charge")

            else:
                raise RuntimeError(
                    f"Unsupported featurization type {featurization}. Supported types are {self._SUPPORTED_FEATURIZATION_TYPES}"
                )

        # if only nuclear charges are embedded no mixing is performed
        self.mixing: Union[nn.Identity, DenseWithCustomDist]
        if self.increase_dim_of_embedded_tensor == 0:
            self.mixing = nn.Identity()
        else:
            self.mixing = DenseWithCustomDist(
                base_embedding_dim + self.increase_dim_of_embedded_tensor,
                base_embedding_dim,
            )

    def forward(self, data: NNPInput) -> torch.Tensor:
        """
        Featurize the input data.

        Parameters
        ----------
        data : NNPInput
            The input data.

        Returns
        -------
        torch.Tensor
            The featurized input data.
        """

        atomic_numbers = data.atomic_numbers
        categorial_embedding = self.atomic_number_embedding(atomic_numbers)

        for additional_embedding in self.embeddings:
            categorial_embedding = additional_embedding(categorial_embedding, data)

        for append_embedding_vector in self.append_to_embedding_tensor:
            categorial_embedding = append_embedding_vector(categorial_embedding, data)

        return self.mixing(categorial_embedding)
