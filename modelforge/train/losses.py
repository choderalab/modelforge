# losses.py

"""
This module contains classes and functions for loss computation and error metrics
for training neural network potentials.
"""

from abc import ABC, abstractmethod
from typing import Dict, List
import torch
from torch import nn
from loguru import logger as log

from modelforge.utils.prop import BatchData

__all__ = [
    "Error",
    "ForceSquaredError",
    "EnergySquaredError",
    "TotalChargeError",
    "PerAtomChargeError",
    "DipoleMomentError",
    "Loss",
    "LossFactory",
    "create_error_metrics",
]


class Error(nn.Module, ABC):
    """
    Abstract base class for error calculation between predicted and true values.
    """

    def __init__(self, scale_by_number_of_atoms: bool = True):
        super().__init__()
        self.scale_by_number_of_atoms = (
            self._scale_by_number_of_atoms
            if scale_by_number_of_atoms
            else lambda error, atomic_subsystem_counts, prefactor=1: error
        )

    @abstractmethod
    def calculate_error(
        self,
        predicted: torch.Tensor,
        true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates the error between the predicted and true values.
        """
        raise NotImplementedError

    @staticmethod
    def calculate_squared_error(
        predicted_tensor: torch.Tensor, reference_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the squared error between the predicted and true values.
        """
        squared_diff = (predicted_tensor - reference_tensor).pow(2)
        error = squared_diff.sum(dim=1, keepdim=True)
        return error

    @staticmethod
    def _scale_by_number_of_atoms(
        error, atomic_counts, prefactor: int = 1
    ) -> torch.Tensor:
        """
        Scales the error by the number of atoms in the atomic subsystems.

        Parameters
        ----------
        error : torch.Tensor
            The error to be scaled.
        atomic_counts : torch.Tensor
            The number of atoms in the atomic subsystems.
        prefactor : int
            Prefactor to adjust for the shape of the property (e.g., vector properties).

        Returns
        -------
        torch.Tensor
            The scaled error.
        """
        scaled_by_number_of_atoms = error / (prefactor * atomic_counts.unsqueeze(1))
        return scaled_by_number_of_atoms


class ForceSquaredError(Error):
    """
    Calculates the per-atom error and aggregates it to per-system mean squared error.
    """

    def calculate_error(
        self,
        per_atom_prediction: torch.Tensor,
        per_atom_reference: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the per-atom squared error."""
        return self.calculate_squared_error(per_atom_prediction, per_atom_reference)

    def forward(
        self,
        per_atom_prediction: torch.Tensor,
        per_atom_reference: torch.Tensor,
        batch: BatchData,
    ) -> torch.Tensor:
        """
        Computes the per-atom error and aggregates it to per-system mean squared error.

        Parameters
        ----------
        per_atom_prediction : torch.Tensor
            The predicted values.
        per_atom_reference : torch.Tensor
            The reference values provided by the dataset.
        batch : BatchData
            The batch data containing metadata and input information.

        Returns
        -------
        torch.Tensor
            The aggregated per-system error.
        """

        # Compute per-atom squared error
        per_atom_squared_error = self.calculate_error(
            per_atom_prediction, per_atom_reference
        )

        # Initialize per-system squared error tensor
        per_system_squared_error = torch.zeros_like(
            batch.metadata.per_system_energy, dtype=per_atom_squared_error.dtype
        )

        # Aggregate error per system
        per_system_squared_error = per_system_squared_error.scatter_add(
            0,
            batch.nnp_input.atomic_subsystem_indices.long().unsqueeze(1),
            per_atom_squared_error,
        )

        # Scale error by number of atoms
        per_system_square_error_scaled = self.scale_by_number_of_atoms(
            per_system_squared_error,
            batch.metadata.atomic_subsystem_counts,
            prefactor=per_atom_prediction.shape[-1],
        )

        return per_system_square_error_scaled.contiguous()


class EnergySquaredError(Error):
    """
    Calculates the per-system mean squared error.
    """

    def calculate_error(
        self,
        per_system_prediction: torch.Tensor,
        per_system_reference: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the per-system squared error."""
        return self.calculate_squared_error(per_system_prediction, per_system_reference)

    def forward(
        self,
        per_system_prediction: torch.Tensor,
        per_system_reference: torch.Tensor,
        batch: BatchData,
    ) -> torch.Tensor:
        """
        Computes the per-system mean squared error.

        Parameters
        ----------
        per_system_prediction : torch.Tensor
            The predicted values.
        per_system_reference : torch.Tensor
            The true values.
        batch : BatchData
            The batch data containing metadata and input information.

        Returns
        -------
        torch.Tensor
            The mean per-system error.
        """

        # Compute per-system squared error
        per_system_squared_error = self.calculate_error(
            per_system_prediction, per_system_reference
        )
        # Scale error by number of atoms
        per_system_square_error_scaled = self.scale_by_number_of_atoms(
            per_system_squared_error,
            batch.metadata.atomic_subsystem_counts,
        )

        return per_system_square_error_scaled


class PerAtomChargeError(Error):
    """
    Calculates the error for per-atom charge.
    """

    """
        Calculates the per-atom error and aggregates it to per-system mean squared error.
        """

    def calculate_error(
        self,
        per_atom_prediction: torch.Tensor,
        per_atom_reference: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the per-atom squared error."""
        return self.calculate_squared_error(per_atom_prediction, per_atom_reference)

    def forward(
        self,
        per_atom_prediction: torch.Tensor,
        per_atom_reference: torch.Tensor,
        batch: BatchData,
    ) -> torch.Tensor:
        """
        Computes the per-atom error and aggregates it to per-system mean squared error.

        Parameters
        ----------
        per_atom_prediction : torch.Tensor
            The predicted values.
        per_atom_reference : torch.Tensor
            The reference values provided by the dataset.
        batch : BatchData
            The batch data containing metadata and input information.

        Returns
        -------
        torch.Tensor
            The aggregated per-system error.
        """

        # Compute per-atom squared error
        per_atom_squared_error = self.calculate_error(
            per_atom_prediction, per_atom_reference
        )

        # Initialize per-system squared error tensor
        per_system_squared_error = torch.zeros_like(
            batch.metadata.per_system_energy, dtype=per_atom_squared_error.dtype
        )

        # Aggregate error per system
        per_system_squared_error = per_system_squared_error.scatter_add(
            0,
            batch.nnp_input.atomic_subsystem_indices.long().unsqueeze(1),
            per_atom_squared_error,
        )

        # Scale error by number of atoms
        per_system_square_error_scaled = self.scale_by_number_of_atoms(
            per_system_squared_error,
            batch.metadata.atomic_subsystem_counts,
            prefactor=1,
        )

        return per_system_square_error_scaled.contiguous()


class TotalChargeError(Error):
    """
    Calculates the error for total charge.
    """

    def calculate_error(
        self,
        total_charge_predict: torch.Tensor,
        total_charge_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the absolute difference between predicted and true total charges.
        """
        error = torch.abs(total_charge_predict - total_charge_true)
        return error  # Shape: [batch_size, 1]

    def forward(
        self,
        total_charge_predict: torch.Tensor,
        total_charge_true: torch.Tensor,
        batch: BatchData,
    ) -> torch.Tensor:
        """
        Computes the error for total charge.

        Parameters
        ----------
        total_charge_predict : torch.Tensor
            The predicted total charges.
        total_charge_true : torch.Tensor
            The true total charges.
        batch : BatchData
            The batch data.

        Returns
        -------
        torch.Tensor
            The error for total charges.
        """
        error = self.calculate_error(total_charge_predict, total_charge_true)
        return error  # No scaling needed


class QuadrupoleMomentError(Error):
    """
    Calculates the error for quadrupole moment.

    Quadrupole moments are represented as 3x3 tensors, so shape [n_systems, 3, 3].
    """

    def calculate_error(
        self,
        quadrupole_predict: torch.Tensor,
        quadrupole_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the squared difference between predicted and true quadrupole moments.
        """
        error = (
            (quadrupole_predict - quadrupole_true).pow(2).sum(dim=(1, 2), keepdim=True)
        )
        return error.squeeze(-1)  # Shape: [n_systems, 1]

    def forward(
        self,
        quadrupole_predict: torch.Tensor,
        quadrupole_true: torch.Tensor,
        batch: BatchData,
    ) -> torch.Tensor:
        """
        Computes the error for quadrupole moment.

        Parameters
        ----------
        quadrupole_predict : torch.Tensor
            The predicted quadrupole moments.
        quadrupole_true : torch.Tensor
            The true quadrupole moments.
        batch : BatchData
            The batch data.

        Returns
        -------
        torch.Tensor
            The error for quadrupole moments.
        """
        error = self.calculate_error(quadrupole_predict, quadrupole_true)
        return error  # No scaling needed


class DipoleMomentError(Error):
    """
    Calculates the error for dipole moment.
    """

    def calculate_error(
        self,
        dipole_predict: torch.Tensor,
        dipole_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the squared difference between predicted and true dipole moments.
        """
        error = (
            (dipole_predict - dipole_true).pow(2).sum(dim=1, keepdim=True)
        )  # Shape: [batch_size, 1]
        return error

    def forward(
        self,
        dipole_predict: torch.Tensor,
        dipole_true: torch.Tensor,
        batch: BatchData,
    ) -> torch.Tensor:
        """
        Computes the error for dipole moment.

        Parameters
        ----------
        dipole_predict : torch.Tensor
            The predicted dipole moments.
        dipole_true : torch.Tensor
            The true dipole moments.
        batch : BatchData
            The batch data.

        Returns
        -------
        torch.Tensor
            The error for dipole moments.
        """
        error = self.calculate_error(dipole_predict, dipole_true)
        return error  # No scaling needed


class Loss(nn.Module):

    _SUPPORTED_PROPERTIES = [
        "per_atom_energy",
        "per_system_energy",
        "per_atom_force",
        "per_system_total_charge",
        "per_system_dipole_moment",
        "per_atom_charge",
        "per_system_quadrupole_moment",
    ]

    def __init__(
        self,
        loss_components: List[str],
        weights_scheduling: Dict[str, torch.Tensor],
    ):
        """
        Calculates the combined loss for energy and force predictions.

        Parameters
        ----------
        loss_components : List[str]
            List of properties to include in the loss calculation.
        weights : Dict[str, float]
            Dictionary containing the weights for each property in the loss calculation.

        Raises
        ------
        NotImplementedError
            If an unsupported loss type is specified.
        """
        super().__init__()
        from torch.nn import ModuleDict

        self.loss_components = loss_components
        self.weights_scheduling = weights_scheduling
        self.loss_functions = ModuleDict()

        for prop in loss_components:
            if prop not in self._SUPPORTED_PROPERTIES:
                raise NotImplementedError(f"Loss type {prop} not implemented.")

            log.info(f"Using loss function for {prop}")
            log.info(
                f"With loss component schedule from weight: {weights_scheduling[prop][0]} to {weights_scheduling[prop][-1]}"
            )

            if prop == "per_atom_force":
                self.loss_functions[prop] = ForceSquaredError(
                    scale_by_number_of_atoms=True
                )
            elif prop == "per_atom_energy":
                self.loss_functions[prop] = EnergySquaredError(
                    scale_by_number_of_atoms=True
                )
            elif prop == "per_system_energy":
                self.loss_functions[prop] = EnergySquaredError(
                    scale_by_number_of_atoms=False
                )
            elif prop == "per_system_total_charge":
                self.loss_functions[prop] = TotalChargeError()
            elif prop == "per_system_dipole_moment":
                self.loss_functions[prop] = DipoleMomentError()
            elif prop == "per_atom_charge":
                self.loss_functions[prop] = PerAtomChargeError()
            elif prop == "per_system_quadrupole_moment":
                self.loss_functions[prop] = QuadrupoleMomentError()
            else:
                raise NotImplementedError(f"Loss type {prop} not implemented.")

            self.register_buffer(prop, self.weights_scheduling[prop])

    def forward(
        self,
        predict_target: Dict[str, torch.Tensor],
        batch: BatchData,
        epoch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculates the combined loss for the specified properties.

        Parameters
        ----------
        predict_target : Dict[str, torch.Tensor]
            Dictionary containing predicted and true values for energy and forces.
        batch : BatchData
            The batch data containing metadata and input information.

        Returns
        -------
        Dict[str, torch.Tensor]
            Individual per-sample loss terms and the combined total loss.
        """
        from modelforge.train.training import (
            _exchange_per_atom_energy_for_per_system_energy,
        )

        # Save the loss as a dictionary
        loss_dict = {}
        # Accumulate loss
        total_loss = torch.zeros_like(batch.metadata.per_system_energy)

        # Iterate over loss properties
        for prop in self.loss_components:
            loss_fn = self.loss_functions[prop]

            prop_ = _exchange_per_atom_energy_for_per_system_energy(prop)
            # NOTE: we always predict per_system_energies, and the dataset
            # also include per_system_energies. If we are normalizing these
            # (indicated by the `per_atom_energy` keyword), we still operate on
            # the per_system_energies but the loss function will divide the
            # error by the number of atoms in the atomic subsystems.
            prop_loss = loss_fn(
                predict_target[f"{prop_}_predict"],
                predict_target[f"{prop_}_true"],
                batch,
            )

            # check that none of the tensors are NaN
            if torch.isnan(prop_loss).any():
                raise ValueError(f"NaN values detected in {prop_} loss.")

            # Accumulate weighted per-sample losses
            weighted_loss = self.weights_scheduling[prop][epoch_idx] * prop_loss

            total_loss += weighted_loss  # Note: total_loss is still per-sample
            loss_dict[prop] = prop_loss  # Store per-sample loss

        # Add total loss to results dict and return
        loss_dict["total_loss"] = total_loss

        return loss_dict


class LossFactory:
    """
    Factory class to create different types of loss functions.
    """

    @staticmethod
    def create_loss(
        loss_components: List[str],
        weights_scheduling: Dict[
            str,
            torch.Tensor,
        ],
    ) -> Loss:
        """
        Creates an instance of the specified loss type.

        Parameters
        ----------
        loss_components : List[str]
            List of properties to include in the loss calculation.
        weights_scheduling : Dict[str, torch.Tensor]
            Dictionary containing the weights for each property in the loss calculation.

        Returns
        -------
        Loss
            An instance of the specified loss function.
        """
        return Loss(
            loss_components,
            weights_scheduling,
        )


from torch.nn import ModuleDict


def create_error_metrics(
    loss_properties: List[str],
    is_loss: bool = False,
) -> ModuleDict:
    """
    Creates a ModuleDict of MetricCollections for the given loss properties.

    Parameters
    ----------
    loss_properties : List[str]
        List of loss properties for which to create the metrics.
    is_loss : bool, optional
        If True, only the loss metric is created, by default False.

    Returns
    -------
    ModuleDict
        A dictionary where keys are loss properties and values are MetricCollections.
    """
    from torchmetrics import MetricCollection
    from torchmetrics.aggregation import MeanMetric
    from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

    if is_loss:
        metric_dict = ModuleDict(
            {prop: MetricCollection([MeanMetric()]) for prop in loss_properties}
        )
        metric_dict["total_loss"] = MetricCollection([MeanMetric()])
    else:
        from modelforge.train.training import (
            _exchange_per_atom_energy_for_per_system_energy,
        )

        # NOTE: we are using the
        # _exchange_per_atom_energy_for_per_system_energy function because, if
        # the `per_atom_energy` loss (i.e., the normalize per_system_energy
        # loss) is used, the validation error is still per_system_energy
        metric_dict = ModuleDict(
            {
                _exchange_per_atom_energy_for_per_system_energy(prop): MetricCollection(
                    [MeanAbsoluteError(), MeanSquaredError(squared=False)]
                )  # only exchange per_atom_energy for per_system_energy
                for prop in loss_properties
            }
        )
    return metric_dict
