"""
This module contains classes and functions for training neural network potentials using PyTorch Lightning.
"""

from typing import Any, Dict, List, Optional, Type, TypeVar, Tuple, Literal
import time
import lightning.pytorch as pL
import torch
from lightning import Trainer
from loguru import logger as log
from openff.units import unit
from torch.nn import ModuleDict
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    CyclicLR,
)

from modelforge.dataset.dataset import DataModule, DatasetParameters
from modelforge.potential.parameters import (
    AimNet2Parameters,
    ANI2xParameters,
    PaiNNParameters,
    PhysNetParameters,
    SAKEParameters,
    SchNetParameters,
    TensorNetParameters,
)
from modelforge.utils.prop import BatchData

T_NNP_Parameters = TypeVar(
    "T_NNP_Parameters",
    ANI2xParameters,
    SAKEParameters,
    SchNetParameters,
    PhysNetParameters,
    PaiNNParameters,
    TensorNetParameters,
    AimNet2Parameters,
)

from modelforge.train.losses import LossFactory, create_error_metrics
from modelforge.train.parameters import RuntimeParameters, TrainingParameters

import matplotlib

matplotlib.use("Agg")

__all__ = [
    "PotentialTrainer",
]


def gradient_norm(model):
    """
    Compute the total gradient norm of a model.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model.

    Returns
    -------
    float
        The total gradient norm.
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def compute_grad_norm(loss, model):
    """
    Compute the gradient norm of the loss with respect to the model parameters.

    Parameters
    ----------
    loss : torch.Tensor
        The loss tensor.
    model : torch.nn.Module
        The neural network model.

    Returns
    -------
    float
        The total gradient norm.
    """
    parameters = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(
        loss.sum(),
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    total_norm = 0.0
    for grad in grads:
        if grad is not None:
            param_norm = grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def _exchange_per_atom_energy_for_per_system_energy(prop: str) -> str:
    """
    Rename 'per_atom_energy' to 'per_system_energy' if applicable.

    Parameters
    ----------
    prop : str
        The property name (e.g., "per_atom_energy").

    Returns
    -------
    str
        The updated property name (e.g., "per_system_energy").
    """
    return "per_system_energy" if prop == "per_atom_energy" else prop


class CalculateProperties(torch.nn.Module):
    _SUPPORTED_PROPERTIES = [
        "per_atom_energy",
        "per_atom_force",
        "per_system_energy",
        "per_system_total_charge",
        "per_system_dipole_moment",
    ]

    def __init__(self, requested_properties: List[str]):
        """
        A utility class for calculating properties such as energies and forces
        from batches using a neural network model.

        Parameters
        ----------
        requested_properties : List[str]
            A list of properties to calculate (e.g., per_atom_energy,
            per_atom_force, per_system_dipole_moment).
        """
        super().__init__()
        self.requested_properties = requested_properties
        self.include_force = "per_atom_force" in self.requested_properties
        self.include_charges = "per_system_total_charge" in self.requested_properties
        # dipole_moment is calculated from charges, thus if dipole moment is requested
        # we also need to calculate charges, even if we don't call per_system_total_charge
        if "per_system_dipole_moment" in self.requested_properties:
            self.include_charges = True

        # Ensure all requested properties are supported
        assert all(
            prop in self._SUPPORTED_PROPERTIES for prop in self.requested_properties
        ), f"Unsupported property requested: {self.requested_properties}"

    @staticmethod
    def _get_forces(
        batch: BatchData,
        model_prediction: Dict[str, torch.Tensor],
        train_mode: bool,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the forces from a given batch using the model.

        Parameters
        ----------
        batch : BatchData
            A single batch of data, including input features and target
            energies.
        model_prediction : Dict[str, torch.Tensor]
            A dictionary containing the predicted energies from the model.
        train_mode : bool
            Whether to retain the graph for gradient computation (True for
            training).
        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing the true and predicted forces.
        """
        nnp_input = batch.nnp_input
        nnp_input.positions.requires_grad_(True)  # Ensure gradients are enabled
        # Cast to float32 and extract true forces
        per_atom_force_true = batch.metadata.per_atom_force.to(torch.float32)

        if per_atom_force_true.numel() < 1:
            raise RuntimeError("No force can be calculated.")

        # Sum the energies before computing the gradient
        total_energy = model_prediction["per_system_energy"].sum()
        # Calculate forces as the negative gradient of energy w.r.t. positions
        grad = torch.autograd.grad(
            total_energy,
            nnp_input.positions,
            create_graph=train_mode,
            retain_graph=train_mode,
            allow_unused=False,
        )[0]

        if grad is None:
            raise RuntimeWarning("Force calculation did not return a gradient")

        per_atom_force_predict = (
            -grad.contiguous()
        )  # Forces are the negative gradient of energy

        return {
            "per_atom_force_true": per_atom_force_true,
            "per_atom_force_predict": per_atom_force_predict,
        }

    @staticmethod
    def _get_energies(
        batch: BatchData,
        model_prediction: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the energies from a given batch using the model.

        Parameters
        ----------
        batch : BatchData
            A single batch of data, including input features and target
            energies.
        model_prediction : Dict[str, torch.Tensor]
            The neural network model used to compute the energies.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing the true and predicted energies.
        """
        per_system_energy_true = batch.metadata.per_system_energy.to(torch.float32)
        per_system_energy_predict = model_prediction["per_system_energy"]

        # Ensure the shapes match
        assert per_system_energy_true.shape == per_system_energy_predict.shape, (
            f"Shapes of true and predicted energies do not match: "
            f"{per_system_energy_true.shape} != {per_system_energy_predict.shape}"
        )
        return {
            "per_system_energy_true": per_system_energy_true,
            "per_system_energy_predict": per_system_energy_predict,
        }

    def _get_charges(
        self,
        batch: BatchData,
        model_prediction: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total molecular charges and dipole moments from the predicted atomic charges.

        Parameters
        ----------
        batch : BatchData
            A batch of data containing input features and target charges.
        model_prediction : Dict[str, torch.Tensor]
            A dictionary containing the predicted charges from the model.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing the true and predicted charges and dipole moments.
        """
        nnp_input = batch.nnp_input
        per_atom_charges_predict = model_prediction[
            "per_atom_charge"
        ]  # Shape: (nr_of_atoms, 1)

        # Calculate predicted total charge by summing per-atom charges for each
        # system
        per_system_total_charge_predict = torch.zeros_like(
            model_prediction["per_system_energy"]
        ).scatter_add_(
            dim=0,
            index=nnp_input.atomic_subsystem_indices.long().unsqueeze(1),
            src=per_atom_charges_predict,
        )  # Shape: [nr_of_systems, 1]

        # Predict the dipole moment
        per_system_dipole_moment = self._predict_dipole_moment(model_prediction, batch)

        return {
            "per_system_total_charge_predict": per_system_total_charge_predict,
            "per_system_total_charge_true": batch.nnp_input.per_system_total_charge,
            "per_system_dipole_moment_predict": per_system_dipole_moment,
            "per_system_dipole_moment_true": batch.metadata.per_system_dipole_moment,
        }

    @staticmethod
    def _predict_dipole_moment(
        model_predictions: Dict[str, torch.Tensor], batch: BatchData
    ) -> torch.Tensor:
        """
        Compute the predicted dipole moment for each system based on the
        predicted partial atomic charges and positions, i.e., the dipole moment
        is calculated as the weighted sum of the partial charges (which requires
        that the coordinates are centered).

        The dipole moment ensures that the predicted charges not only sum up to
        the correct total charge but also reproduce the reference dipole moment.

        Parameters
        ----------
        model_predictions : Dict[str, torch.Tensor]
            A dictionary containing the predicted atomic charges from the model.
        batch : BatchData
            A batch of data containing the atomic positions and indices.

        Returns
        -------
        torch.Tensor
            The predicted dipole moment for each system.
        """
        per_atom_charge = model_predictions["per_atom_charge"]  # Shape: [num_atoms, 1]
        positions = batch.nnp_input.positions  # Shape: [num_atoms, 3]
        per_atom_charge = per_atom_charge  # Shape: [num_atoms, 1]
        per_atom_dipole_contrib = per_atom_charge * positions  # Shape: [num_atoms, 3]

        indices = batch.nnp_input.atomic_subsystem_indices.long()  # Shape: [num_atoms]
        indices = indices.unsqueeze(-1).expand(-1, 3)  # Shape: [num_atoms, 3]

        # Calculate dipole moment as the sum of dipole contributions for each
        # system
        dipole_predict = torch.zeros(
            (model_predictions["per_system_energy"].shape[0], 3),
            device=positions.device,
            dtype=positions.dtype,
        ).scatter_add_(
            dim=0,
            index=indices,
            src=per_atom_dipole_contrib,
        )  # Shape: [nr_of_systems, 3]

        return dipole_predict

    def forward(
        self,
        batch: BatchData,
        model: torch.nn.Module,
        train_mode: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes energies, forces, and charges from a given batch using the
        model.

        Parameters
        ----------
        batch : BatchData
            A single batch of data, including input features and target
            energies.
        model : Type[torch.nn.Module]
            The neural network model used to compute the properties.
        train_mode : bool, optional
            Whether to calculate gradients for forces (default is False).

        Returns
        -------
        Dict[str, torch.Tensor]
            The true and predicted energies and forces from the dataset and the
            model.
        """
        predict_target = {}
        nnp_input = batch.nnp_input
        model_prediction = model.forward(nnp_input)

        # Get predicted energies
        energies = self._get_energies(batch, model_prediction)
        predict_target.update(energies)

        # Get forces if they are included in the requested properties
        if self.include_force:
            forces = self._get_forces(batch, model_prediction, train_mode)
            predict_target.update(forces)

        # Get charges if they are included in the requested properties
        if self.include_charges:
            charges = self._get_charges(batch, model_prediction)
            predict_target.update(charges)

        return predict_target


class TrainingAdapter(pL.LightningModule):
    """
    A Lightning module that encapsulates the training process for neural network potentials.
    """

    def __init__(
        self,
        *,
        potential_parameter: T_NNP_Parameters,
        dataset_statistic: Dict[str, Dict[str, unit.Quantity]],
        training_parameter: TrainingParameters,
        optimizer_class: Type[Optimizer],
        nr_of_training_batches: int = -1,
        potential_seed: Optional[int] = None,
    ):
        """
        Initialize the TrainingAdapter with model and training configuration.

        Parameters
        ----------
        potential_parameter : T_NNP_Parameters
            Parameters for the potential model.
        dataset_statistic : Dict[str, Dict[str, unit.Quantity]]
            Dataset statistics such as mean and standard deviation.
        training_parameter : TrainingParameters
            Training configuration, including optimizer, learning rate, and loss functions.
        optimizer_class : Type[Optimizer]
            The optimizer class to use for training.
        nr_of_training_batches : int, optional
            Number of training batches (default is -1).
        potential_seed : Optional[int], optional
            Seed for initializing the model (default is None).
        """
        from modelforge.potential.potential import setup_potential

        self.epoch_start_time = None

        super().__init__()
        self.save_hyperparameters()
        self.training_parameter = training_parameter

        # Setup the potential model
        self.potential = setup_potential(
            potential_parameter=potential_parameter,
            dataset_statistic=dataset_statistic,
            potential_seed=potential_seed,
            jit=False,
            use_training_mode_neighborlist=True,
        )

        # Determine which properties to include based on loss components
        self.include_force = (
            "per_atom_force" in training_parameter.loss_parameter.loss_components
        )

        # Initialize the property calculation utility
        self.calculate_predictions = CalculateProperties(
            training_parameter.loss_parameter.loss_components
        )
        self.optimizer_class = optimizer_class
        self.learning_rate = training_parameter.lr
        self.lr_scheduler = training_parameter.lr_scheduler

        # Setup logging flags based on verbosity
        self.log_histograms = training_parameter.verbose
        self.log_on_training_step = training_parameter.verbose

        # Initialize the loss function with scheduled weights
        weights_scheduling = self._setup_weights_scheduling(
            training_parameter=training_parameter,
        )
        self.loss = LossFactory.create_loss(
            loss_components=training_parameter.loss_parameter.loss_components,
            weights_scheduling=weights_scheduling,
        )

        # Initialize performance metrics for different phases
        self.test_metrics = create_error_metrics(
            training_parameter.loss_parameter.loss_components
        )
        self.val_metrics = create_error_metrics(
            training_parameter.loss_parameter.loss_components
        )
        self.train_metrics = create_error_metrics(
            training_parameter.loss_parameter.loss_components
        )

        self.loss_metrics = create_error_metrics(
            training_parameter.loss_parameter.loss_components, is_loss=True
        )

        # Initialize dictionaries to store predictions and targets
        self.train_preds: Dict[str, Dict[int, torch.Tensor]] = {
            "energy": {},
            "force": {},
        }
        self.train_targets: Dict[str, Dict[int, torch.Tensor]] = {
            "energy": {},
            "force": {},
        }
        self.val_preds: Dict[str, Dict[int, torch.Tensor]] = {
            "energy": {},
            "force": {},
        }
        self.val_targets: Dict[str, Dict[int, torch.Tensor]] = {
            "energy": {},
            "force": {},
        }
        self.test_preds: Dict[str, Dict[int, torch.Tensor]] = {
            "energy": {},
            "force": {},
        }
        self.test_targets: Dict[str, Dict[int, torch.Tensor]] = {
            "energy": {},
            "force": {},
        }

        # Initialize indices for validation and testing NOTE: this indices map
        # back to the dataset
        self.val_indices: Dict[int, torch.Tensor] = {}
        self.test_indices: Dict[int, torch.Tensor] = {}
        self.train_indices: Dict[int, torch.Tensor] = {}

        # Track outlier errors over epochs
        self.outlier_errors_over_epochs: Dict[str, int] = {}
        self.number_of_training_batches = nr_of_training_batches

    def _setup_weights_scheduling(
        self, training_parameter: TrainingParameters
    ) -> Dict[str, torch.Tensor]:
        """
        Setup weight scheduling for loss components over epochs.

        Parameters
        ----------
        training_parameter : TrainingParameters
            The training configuration.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary mapping loss component names to their scheduled
            weights.
        """

        weights_scheduling: Dict[str, torch.Tensor] = {}
        initial_weights = training_parameter.loss_parameter.weight
        nr_of_epochs = training_parameter.number_of_epochs

        for key, initial_weight in initial_weights.items():
            target_weight = training_parameter.loss_parameter.target_weight[key]
            mixing_steps = training_parameter.loss_parameter.mixing_steps[key]

            # Create a linear schedule from initial to target weight
            mixing_scheme = torch.arange(
                start=initial_weight,
                end=target_weight,
                step=mixing_steps,
            )
            assert (
                len(mixing_scheme) < nr_of_epochs
            ), "The number of epochs is less than the number of steps in the weight scheduling"

            # Fill up the rest of the epochs with the target weight
            weights_scheduling[key] = torch.cat(
                [
                    mixing_scheme,
                    torch.ones(nr_of_epochs - mixing_scheme.shape[0]) * target_weight,
                ]
            )
            assert (
                weights_scheduling[key].shape[0] == nr_of_epochs
            ), "Weight scheduling length mismatch."
        return weights_scheduling

    def forward(self, batch: BatchData) -> Dict[str, torch.Tensor]:
        """
        Forward pass to compute energies, forces, and other properties from a
        batch.

        Parameters
        ----------
        batch : BatchData
            A batch of data including input features and target properties.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of predicted properties (energies, forces, etc.).
        """
        return self.potential(batch)

    def config_prior(self):
        """
        Configures model-specific priors if the model implements them.
        """
        if hasattr(self.potential, "_config_prior"):
            return self.potential._config_prior()

        log.warning("Model does not implement _config_prior().")
        raise NotImplementedError()

    @staticmethod
    def _update_metrics(
        metrics: ModuleDict,
        predict_target: Dict[str, torch.Tensor],
    ):
        """
        Updates the provided metric collections with the predicted and true
        targets.

        Parameters
        ----------
        metrics : ModuleDict
            Metric collections for energy and force evaluation.
        predict_target : Dict[str, torch.Tensor]
            Dictionary containing predicted and true values for properties.
        """

        for prop, metric_collection in metrics.items():
            prop = _exchange_per_atom_energy_for_per_system_energy(
                prop
            )  # only exchange per_atom_energy for per_system_energy
            preds = predict_target[f"{prop}_predict"].detach()
            targets = predict_target[f"{prop}_true"].detach()
            metric_collection.update(preds, targets)

    def on_validation_epoch_start(self):
        """Reset validation metrics at the start of the validation epoch."""
        self._reset_metrics(self.val_metrics)

    def on_test_epoch_start(self):
        """Reset test metrics at the start of the test epoch."""
        self._reset_metrics(self.test_metrics)

    def _reset_metrics(self, metrics: ModuleDict):
        """Utility function to reset all metrics in a ModuleDict."""
        for metric_collection in metrics.values():
            for metric in metric_collection.values():
                metric.reset()

    def training_step(
        self,
        batch: BatchData,
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Training step to compute the MSE loss for a given batch.

        Parameters
        ----------
        batch : BatchData
            The batch of data provided for the training.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        torch.Tensor
            The loss tensor computed for the current training step.
        """

        # Calculate predictions based on the current batch
        predict_target = self.calculate_predictions(
            batch, self.potential, self.training
        )

        # Compute loss using the loss factory
        loss_dict = self.loss(
            predict_target,
            batch,
            self.current_epoch,
        )  # Contains per-sample losses

        # Update loss metrics with per-sample losses
        batch_size = batch.batch_size()
        for key, metric in loss_dict.items():
            self.loss_metrics[key].update(metric.detach(), batch_size=batch_size)

            # Compute and log gradient norms for each loss component
            if self.training_parameter.log_norm:
                if key == "total_loss":
                    continue  # Skip total loss for gradient norm logging
                grad_norm = compute_grad_norm(metric.mean(), self)
                self.log(f"grad_norm/{key}", grad_norm, sync_dist=True)

        # Save energy predictions and targets
        self._update_predictions(
            predict_target,
            self.train_preds,
            self.train_targets,
            self.train_indices,
            batch_idx,
            batch,
        )

        # Compute the mean loss for optimization
        total_loss = loss_dict["total_loss"].mean()
        return total_loss

    def validation_step(self, batch: BatchData, batch_idx: int) -> None:
        """
        Validation step to compute validation loss and metrics.
        """

        # Ensure positions require gradients for force calculation
        batch.nnp_input.positions.requires_grad_(True)
        with torch.set_grad_enabled(True):
            # calculate energy and forces
            predict_target = self.calculate_predictions(
                batch, self.potential, self.potential.training
            )

        # Update validation metrics
        self._update_metrics(self.val_metrics, predict_target)

        # Save energy predictions and targets
        self._update_predictions(
            predict_target,
            self.val_preds,
            self.val_targets,
            self.val_indices,
            batch_idx,
            batch,
        )

    def test_step(self, batch: BatchData, batch_idx: int) -> None:
        """
        Test step to compute the test loss and metrics.
        """
        # Ensure positions require gradients for force calculation
        batch.nnp_input.positions.requires_grad_(True)
        with torch.set_grad_enabled(True):
            # calculate energy and forces
            predict_target = self.calculate_predictions(
                batch, self.potential, self.training
            )
        # Update and log metrics
        self._update_metrics(self.test_metrics, predict_target)

        # Save energy predictions and targets
        self._update_predictions(
            predict_target,
            self.test_preds,
            self.test_targets,
            self.test_indices,
            batch_idx,
            batch,
        )

    def _update_predictions(
        self,
        predict_target: Dict[str, torch.Tensor],
        preds: Dict[str, Dict[int, torch.Tensor]],
        targets: Dict[str, Dict[int, torch.Tensor]],
        indices: Dict[int, torch.Tensor],
        batch_idx: int,
        batch: BatchData,
    ):
        """
        Update the predictions and targets dictionaries with the provided data.

        Parameters
        ----------
        predict_target : Dict[str, torch.Tensor]
            The predicted and true values for properties.
        preds : Dict[str, Dict[int, torch.Tensor]]
            Dictionary to store predictions.
        targets : Dict[str, Dict[int, torch.Tensor]]
            Dictionary to store targets.
        indices : Dict[int, torch.Tensor]
            Dictionary to store indices referencing the dataset.
        batch_idx : int
            The index of the current batch.
        batch : BatchData
            The current batch of data.
        """
        # Update energy predictions and targets
        preds["energy"].update(
            {batch_idx: predict_target["per_system_energy_predict"].detach().cpu()}
        )
        targets["energy"].update(
            {batch_idx: predict_target["per_system_energy_true"].detach().cpu()}
        )
        # Save dataset indices
        indices.update(
            {
                batch_idx: batch.metadata.atomic_subsystem_indices_referencing_dataset.detach().cpu()
            }
        )
        # Save force predictions and targets if forces are included
        if "per_atom_force_predict" in predict_target:
            preds["force"].update(
                {batch_idx: predict_target["per_atom_force_predict"].detach().cpu()}
            )
            targets["force"].update(
                {batch_idx: predict_target["per_atom_force_true"].detach().cpu()}
            )

    def _get_energy_tensors(
        self,
        preds: Dict[int, torch.Tensor],
        targets: Dict[int, torch.Tensor],
        indices: Dict[int, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """
        Gathers and pads prediction and target tensors across processes.

        Parameters
        ----------
        preds : Dict[int, torch.Tensor]
            Dictionary of predictions from different batches.
        targets : Dict[int, torch.Tensor]
            Dictionary of targets from different batches.
        indices : Dict[int, torch.Tensor]
            Dictionary of indices from different batches.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]
            Gathered predictions, targets, indices, maximum length, and padding size.
        """
        # Concatenate the tensors
        preds_tensor = torch.cat(list(preds.values()))
        targets_tensor = torch.cat(list(targets.values()))
        indices_tensor = torch.cat(list(indices.values())).unique()

        # Get maximum length across all processes
        local_length = torch.tensor([preds_tensor.size(0)], device=preds_tensor.device)
        max_length = int(self.all_gather(local_length).max())

        pad_size = max_length - preds_tensor.size(0)
        if pad_size > 0:
            log.debug(f"Padding tensors to the same length: {max_length}")
            log.debug(f"Triggered at device: {self.global_rank}")
            preds_tensor = torch.nn.functional.pad(preds_tensor, (0, pad_size))
            targets_tensor = torch.nn.functional.pad(targets_tensor, (0, pad_size))

        # Gather across processes
        gathered_preds = self.all_gather(preds_tensor)
        gathered_targets = self.all_gather(targets_tensor)
        gathered_indices = self.all_gather(indices_tensor)

        return gathered_preds, gathered_targets, gathered_indices, max_length, pad_size

    def _get_force_tensors(
        self, preds: Dict[int, torch.Tensor], targets: Dict[int, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Gathers and pads force prediction and target tensors across processes.

        Parameters
        ----------
        preds : Dict[int, torch.Tensor]
            Dictionary of force predictions from different batches.
        targets : Dict[int, torch.Tensor]
            Dictionary of force targets from different batches.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, int, int]
            Gathered force predictions, targets, maximum length, and padding size.
        """
        # Concatenate the tensors
        preds_tensor = torch.cat(list(preds.values()))
        targets_tensor = torch.cat(list(targets.values()))

        # Get maximum length across all processes
        local_length = torch.tensor([preds_tensor.size(0)], device=preds_tensor.device)
        max_length = int(self.all_gather(local_length).max())

        pad_size = max_length - preds_tensor.size(0)
        if pad_size > 0:
            log.debug(f"Padding force tensors to the same length: {max_length}")
            log.debug(f"Triggered at device: {self.global_rank}")
            # For forces, pad the last dimension (x, y, z)
            preds_tensor = torch.nn.functional.pad(preds_tensor, (0, 0, 0, pad_size))
            targets_tensor = torch.nn.functional.pad(
                targets_tensor, (0, 0, 0, pad_size)
            )
        # Gather across processes
        gathered_preds = self.all_gather(preds_tensor)
        gathered_targets = self.all_gather(targets_tensor)

        return gathered_preds, gathered_targets, max_length, pad_size

    def _log_force_errors(
        self,
        preds: Dict[str, Dict[int, torch.Tensor]],
        targets: Dict[str, Dict[int, torch.Tensor]],
        indices: Dict[int, torch.Tensor],
        phase: str,
    ):
        """
        Log the force error statistics as histograms.

        Parameters
        ----------
        preds : Dict[str, Dict[int, torch.Tensor]]
            Dictionary of force predictions.
        targets : Dict[str, Dict[int, torch.Tensor]]
            Dictionary of force targets.
        indices : Dict[int, torch.Tensor]
            Dictionary of indices referencing the dataset.
        phase : str
            The phase name ('train', 'val', or 'test').
        """

        # Gather tensors
        gathered_preds, gathered_targets, max_length, pad_size = (
            self._get_force_tensors(
                preds["force"],
                targets["force"],
            )
        )

        if self.global_rank == 0:
            # Remove padding
            total_length = max_length * self.trainer.world_size
            gathered_preds = gathered_preds.reshape(total_length, 3)[
                : total_length - pad_size * self.trainer.world_size
            ]
            gathered_targets = gathered_targets.reshape(total_length, 3)[
                : total_length - pad_size * self.trainer.world_size
            ]
            errors = gathered_targets - gathered_preds
            errors_magnitude = errors.norm(dim=1)  # Compute magnitude of force errors
            # make sure that errors are finite
            assert torch.all(
                torch.isfinite(errors_magnitude)
            ), "Force errors contain NaN or Inf values."

            # Create histogram
            histogram_fig = self._create_error_histogram(
                errors_magnitude,
                title=f"{phase.capitalize()} Magnitude of Force Error Histogram - Epoch {self.current_epoch}",
            )

            self._log_plots(phase, None, histogram_fig, force=True)

    def _log_plots(self, phase: str, regression_fig, histogram_fig, force=False):
        """
        Log the regression and error histogram plots for the given phase.

        Parameters
        ----------
        phase : str
            The phase name ('train', 'val', or 'test').
        regression_fig : matplotlib.figure.Figure
            The regression plot figure.
        histogram_fig : matplotlib.figure.Figure
            The error histogram figure.
        force : bool, optional
            Whether to indicate force-related plots (default is False).

        Returns
        -------
        None
        """

        logger_name = self.training_parameter.experiment_logger.logger_name.lower()
        plot_frequency = (
            self.training_parameter.plot_frequency
        )  # how often to log plots

        if logger_name == "wandb":
            import wandb

            # Log only every nth epoch for validation, but always log for test
            if phase == "test" or self.current_epoch % plot_frequency == 0:
                # NOTE: only log every nth epoch for validation, but always log
                # for test
                if not force and regression_fig is not None:
                    # Log histogram of errors and regression plot
                    self.logger.experiment.log(
                        {f"{phase}/regression_plot": wandb.Image(regression_fig)},
                        # step=self.current_epoch
                    )
                self.logger.experiment.log(
                    {
                        f"{phase}/{'force_' if force else 'energy_'}error_histogram": wandb.Image(
                            histogram_fig
                        )
                    },
                    # step=self.current_epoch
                )

        elif logger_name == "tensorboard":
            # Similar adjustments for tensorboard
            if phase == "test" or self.current_epoch % plot_frequency == 0:
                if not force and regression_fig is not None:
                    self.logger.experiment.add_figure(
                        f"{phase}_regression_plot_epoch_{self.current_epoch}",
                        regression_fig,
                        self.current_epoch,
                    )
                self.logger.experiment.add_figure(
                    f"{phase}_{'force_' if force else ''}error_histogram_epoch_{self.current_epoch}",
                    histogram_fig,
                    self.current_epoch,
                )
        else:
            log.warning(f"No logger found to log {phase} plots")

        import matplotlib.pyplot as plt

        # Close the figures
        if regression_fig is not None:
            plt.close(regression_fig)
        plt.close(histogram_fig)

    def _create_regression_plot(
        self, targets: torch.Tensor, predictions: torch.Tensor, title="Regression Plot"
    ):
        """
        Creates a regression plot comparing true targets and predictions.

        Parameters
        ----------
        targets : torch.Tensor
            Array of true target values.
        predictions : torch.Tensor
            Array of predicted values.
        title : str, optional
            Title of the plot. Default is 'Regression Plot'.

        Returns
        -------
        matplotlib.figure.Figure
            The regression plot figure.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        targets = targets.cpu().numpy()
        predictions = predictions.cpu().numpy()
        ax.scatter(targets, predictions, alpha=0.5)
        ax.plot([targets.min(), targets.max()], [targets.min(), targets.max()], "r--")
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(title)
        return fig

    def _create_error_histogram(self, errors: torch.Tensor, title="Error Histogram"):
        """
        Create an error histogram plot.

        Parameters
        ----------
        errors : torch.Tensor
            Tensor of error magnitudes.
        title : str, optional
            Title of the histogram (default is 'Error Histogram').

        Returns
        -------
        matplotlib.figure.Figure
            The error histogram figure.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        errors_np = errors.cpu().numpy().flatten()

        # Compute mean and standard deviation
        mean_error = np.mean(errors_np)
        std_error = np.std(errors_np)

        fig, ax = plt.subplots(figsize=(8, 6))
        bins = 50

        # Plot histogram and get bin data
        counts, bin_edges, patches = ax.hist(
            errors_np, bins=bins, alpha=0.75, edgecolor="black"
        )

        # Set y-axis to log scale
        ax.set_yscale("log")

        # Highlight outlier bins beyond 3 standard deviations
        for count, edge_left, edge_right, patch in zip(
            counts, bin_edges[:-1], bin_edges[1:], patches
        ):
            if (edge_left < mean_error - 3 * std_error) or (
                edge_right > mean_error + 3 * std_error
            ):
                patch.set_facecolor("red")
            else:
                patch.set_facecolor("blue")

        # Add vertical lines for mean and standard deviations
        ax.axvline(mean_error, color="k", linestyle="dashed", linewidth=1, label="Mean")
        ax.axvline(
            mean_error + 3 * std_error,
            color="r",
            linestyle="dashed",
            linewidth=1,
            label="±3 Std Dev",
        )
        ax.axvline(
            mean_error - 3 * std_error, color="r", linestyle="dashed", linewidth=1
        )

        ax.set_xlabel("Error")
        ax.set_ylabel("Frequency (Log Scale)")
        ax.set_title(title)
        ax.legend()

        return fig

    def _log_figures_for_each_phase(
        self,
        preds: Dict[str, Dict[int, torch.Tensor]],
        targets: Dict[str, Dict[int, torch.Tensor]],
        indices: Dict[int, torch.Tensor],
        phase: Literal["train", "val", "test"],
    ):
        """
        Log regression plots and error histograms for a specific phase.

        Parameters
        ----------
        preds : Dict[str, Dict[int, torch.Tensor]]
            Dictionary of predictions.
        targets : Dict[str, Dict[int, torch.Tensor]]
            Dictionary of targets.
        indices : Dict[int, torch.Tensor]
            Dictionary of dataset indices.
        phase : Literal["train", "val", "test"]
            The phase name.

        """
        # Gather across processes
        gathered_preds, gathered_targets, gathered_indices, max_length, pad_size = (
            self._get_energy_tensors(
                preds["energy"],
                targets["energy"],
                indices,
            )
        )

        # Proceed only on main process
        if self.global_rank == 0:
            # Remove padding
            total_length = max_length * self.trainer.world_size
            gathered_preds = gathered_preds.reshape(total_length)[
                : total_length - pad_size * self.trainer.world_size
            ]
            gathered_targets = gathered_targets.reshape(total_length)[
                : total_length - pad_size * self.trainer.world_size
            ]
            gathered_indices = gathered_indices.reshape(total_length)[
                : total_length - pad_size * self.trainer.world_size
            ]

            errors = gathered_targets - gathered_preds
            if errors.size == 0:
                log.warning("Errors array is empty.")

            # Create regression plot
            regression_fig = self._create_regression_plot(
                gathered_targets,
                gathered_preds,
                title=f"{phase.capitalize()} Regression Plot - Epoch {self.current_epoch}",
            )

            # Generate error histogram plot
            histogram_fig = self._create_error_histogram(
                errors,
                title=f"{phase.capitalize()} Error Histogram - Epoch {self.current_epoch}",
            )
            self._log_plots(phase, regression_fig, histogram_fig)

            # Log outlier error counts for non-training phases
            if phase != "train":
                self._identify__and_log_top_k_errors(errors, gathered_indices, phase)
                self.log_dict(
                    self.outlier_errors_over_epochs, on_epoch=True, rank_zero_only=True
                )

    def _identify__and_log_top_k_errors(
        self,
        errors: torch.Tensor,
        indices: torch.Tensor,
        phase: Literal["train", "val", "test"],
        k: int = 3,
    ):
        """
        Identify and log the top k largest errors.

        Parameters
        ----------
        errors : torch.Tensor
            Tensor of error magnitudes.
        indices : torch.Tensor
            Tensor of dataset indices corresponding to the errors.
        phase : Literal["train", "val", "test"]
            The phase name.
        k : int, optional
            Number of top errors to track (default is 3).

        """

        # Compute absolute errors
        abs_errors = torch.abs(errors).detach().cpu()
        # Flatten tensors
        abs_errors = abs_errors.flatten()
        indices = indices.flatten().long().detach().cpu()

        # Get top k errors and their corresponding indices
        top_k_errors, top_k_indices = torch.topk(abs_errors, k)

        top_k_indices = indices[top_k_indices].tolist()
        for idx, error in zip(top_k_indices, top_k_errors.tolist()):
            key = f"outlier_count/{phase}/{idx}"
            if key not in self.outlier_errors_over_epochs:
                self.outlier_errors_over_epochs[key] = 0
            self.outlier_errors_over_epochs[key] += 1
            log.info(
                f"{self.current_epoch}: {phase} : Outlier error {error} at index {idx}."
            )

    def _clear_error_tracking(self, preds, targets, incides):
        """
        Clear the prediction, target, and index tracking dictionaries.

        Parameters
        ----------
        preds : Dict[str, Dict[int, torch.Tensor]]
            Dictionary of predictions.
        targets : Dict[str, Dict[int, torch.Tensor]]
            Dictionary of targets.
        indices : Dict[int, torch.Tensor]
            Dictionary of dataset indices.

        """
        for d in [preds, targets]:
            d["energy"].clear()
            d["force"].clear()
        incides.clear()

    def on_test_epoch_end(self):
        """Logs metrics and figures at the end of the test epoch."""
        self._log_metrics(self.test_metrics, "test")
        self._log_figures_for_each_phase(
            self.test_preds,
            self.test_targets,
            self.test_indices,
            "test",
        )
        # Clear the dictionaries after logging
        self._clear_error_tracking(
            self.test_preds,
            self.test_targets,
            self.test_indices,
        )

    def on_validation_epoch_end(self):
        """Logs metrics and figures at the end of the validation epoch."""
        self._log_metrics(self.val_metrics, "val")
        self._log_figures_for_each_phase(
            self.val_preds,
            self.val_targets,
            self.val_indices,
            "val",
        )
        # Clear the dictionaries after logging
        self._clear_error_tracking(
            self.val_preds,
            self.val_targets,
            self.val_indices,
        )

    def on_train_epoch_start(self):
        """Start the epoch timer."""
        self.epoch_start_time = time.time()

    def _log_time(self):
        """Log the time taken per epoch to W&B."""
        epoch_time = time.time() - self.epoch_start_time
        if isinstance(self.logger, pL.loggers.WandbLogger):
            # Log epoch duration to W&B
            self.logger.experiment.log(
                {"epoch_time": epoch_time, "epoch": self.current_epoch}
            )
        else:
            log.warning("Weights & Biases logger not found; epoch time not logged.")

    def on_train_epoch_end(self):
        """Logs metrics, learning rate, histograms, and figures at the end of the training epoch."""
        self._log_metrics(self.loss_metrics, "loss")
        # this performs gather operations and logs only at rank == 0
        self._log_figures_for_each_phase(
            self.train_preds,
            self.train_targets,
            self.train_indices,
            "train",
        )
        if self.include_force:
            self._log_force_errors(
                self.train_preds,
                self.train_targets,
                self.train_indices,
                "train",
            )
        # Clear the dictionaries after logging
        self._clear_error_tracking(
            self.train_preds,
            self.train_targets,
            self.train_indices,
        )

        self._log_learning_rate()
        self._log_time()
        self._log_histograms()
        # log the weights of the different loss components
        if self.trainer.is_global_zero:
            for key, weight in self.loss.weights_scheduling.items():
                self.log(
                    f"loss/{key}/weight",
                    weight[self.current_epoch],
                    rank_zero_only=True,
                )

    def _log_learning_rate(self):
        """Logs the current learning rate."""
        sch = self.lr_schedulers()
        if self.trainer.is_global_zero:
            try:
                self.log(
                    "lr",
                    sch.get_last_lr()[0],
                    on_epoch=True,
                    prog_bar=True,
                    rank_zero_only=True,
                )
            except AttributeError:
                pass

    def _log_metrics(self, metrics: ModuleDict, phase: str):
        """
        Log all accumulated metrics for a given phase.

        Parameters
        ----------
        metrics : ModuleDict
            The metrics to log.
        phase : str
            The phase name ('train', 'val', or 'test').

        """
        # abbreviate long names to shorter versions
        abbreviate = {
            "MeanAbsoluteError": "mae",
            "MeanSquaredError": "rmse",
            "MeanMetric": "mse",  # NOTE: MeanMetric is the MSE since we accumulate the squared error
        }  # NOTE: MeanSquaredError(squared=False) is RMSE

        for prop, metric_collection in metrics.items():
            for metric_name, metric in metric_collection.items():
                metric_value = metric.compute()
                metric.reset()
                self.log(
                    f"{phase}/{prop}/{abbreviate[metric_name]}",
                    metric_value,
                    prog_bar=True,
                    sync_dist=True,
                )

    def _log_histograms(self):
        """
        Log histograms of model parameters and their gradients if enabled.
        """
        if self.log_histograms:
            for name, params in self.named_parameters():
                if params is not None:
                    self.logger.experiment.add_histogram(
                        name, params, self.current_epoch
                    )
                if params.grad is not None:
                    self.logger.experiment.add_histogram(
                        f"{name}.grad", params.grad, self.current_epoch
                    )

    def configure_optimizers(self):
        """
        Configure the optimizers and learning rate schedulers.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the optimizer and optionally the scheduler.
        """
        from modelforge.train.parameters import (
            ReduceLROnPlateauConfig,
            CosineAnnealingLRConfig,
            CosineAnnealingWarmRestartsConfig,
            OneCycleLRConfig,
            CyclicLRConfig,
        )

        # Separate parameters into weight and bias groups
        weight_params = []
        bias_params = []

        for name, param in self.potential.named_parameters():
            if (
                "weight" in name
                or "atomic_shift" in name
                or "gate" in name
                or "agh" in name
            ):
                weight_params.append(param)
            elif "bias" in name or "atomic_scale" in name:
                bias_params.append(param)
            else:
                # If parameter type is unknown, raise an error
                raise ValueError(f"Unknown parameter type: {name}")

        # Define parameter groups with different weight decay
        param_groups = [
            {
                "params": weight_params,
                "lr": self.learning_rate,
                "weight_decay": 1e-3,  # Apply weight decay to weights
            },
            {
                "params": bias_params,
                "lr": self.learning_rate,
                "weight_decay": 0.0,  # No weight decay for biases
            },
        ]

        optimizer = torch.optim.AdamW(param_groups)

        lr_scheduler_config = self.lr_scheduler

        if lr_scheduler_config is None:
            return {"optimizer": optimizer}

        interval = lr_scheduler_config.interval
        frequency = lr_scheduler_config.frequency
        monitor = (
            lr_scheduler_config.monitor or self.monitor
        )  # Use default monitor if not specified

        # Determine the scheduler class and parameters
        if isinstance(lr_scheduler_config, ReduceLROnPlateauConfig):
            scheduler_class = ReduceLROnPlateau
            scheduler_params = lr_scheduler_config.model_dump(
                exclude={"scheduler_name", "frequency", "interval", "monitor"}
            )
        elif isinstance(lr_scheduler_config, CosineAnnealingLRConfig):
            scheduler_class = CosineAnnealingLR
            scheduler_params = lr_scheduler_config.model_dump(
                exclude={"scheduler_name", "frequency", "interval", "monitor"}
            )
        elif isinstance(lr_scheduler_config, CosineAnnealingWarmRestartsConfig):
            scheduler_class = CosineAnnealingWarmRestarts
            scheduler_params = lr_scheduler_config.model_dump(
                exclude={"scheduler_name", "frequency", "interval", "monitor"}
            )
        elif isinstance(lr_scheduler_config, OneCycleLRConfig):
            scheduler_class = OneCycleLR
            scheduler_params = lr_scheduler_config.model_dump(
                exclude={
                    "scheduler_name",
                    "frequency",
                    "interval",
                    "monitor",
                    "steps_per_epoch",
                    "total_steps",
                }
            )

            # Calculate steps_per_epoch
            steps_per_epoch = self.number_of_training_batches
            scheduler_params["steps_per_epoch"] = steps_per_epoch
            scheduler_params["epochs"] = lr_scheduler_config.epochs
        elif isinstance(lr_scheduler_config, CyclicLRConfig):
            scheduler_class = CyclicLR
            scheduler_params = lr_scheduler_config.model_dump(
                exclude={
                    "scheduler_name",
                    "frequency",
                    "interval",
                    "monitor",
                    "epochs_up",
                    "epochs_down",
                }
            )

            # Calculate steps_per_epoch
            steps_per_epoch = self.number_of_training_batches

            # Calculate step_size_up and step_size_down
            epochs_up = lr_scheduler_config.epochs_up
            epochs_down = (
                lr_scheduler_config.epochs_down or epochs_up
            )  # Symmetric cycle if not specified
            step_size_up = int(epochs_up * steps_per_epoch)
            step_size_down = int(epochs_down * steps_per_epoch)

            scheduler_params["step_size_up"] = step_size_up
            scheduler_params["step_size_down"] = step_size_down
        else:
            raise NotImplementedError(
                f"Unsupported learning rate scheduler: {lr_scheduler_config.scheduler_name}"
            )
        lr_scheduler_instance = scheduler_class(optimizer, **scheduler_params)

        scheduler = {
            "scheduler": lr_scheduler_instance,
            "monitor": monitor,  # Name of the metric to monitor
            "interval": interval,
            "frequency": frequency,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


from openff.units import unit


class PotentialTrainer:
    """
    Class for training neural network potentials using PyTorch Lightning.
    """

    def __init__(
        self,
        *,
        dataset_parameter: DatasetParameters,
        potential_parameter: T_NNP_Parameters,
        training_parameter: TrainingParameters,
        runtime_parameter: RuntimeParameters,
        dataset_statistic: Dict[str, Dict[str, unit.Quantity]],
        use_default_dataset_statistic: bool,
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        potential_seed: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Initializes the TrainingAdapter with the specified model and training
        configuration.

        Parameters
        ----------
        dataset_parameter : DatasetParameters
            Parameters for the dataset.
        potential_parameter : Union[ANI2xParameters, SAKEParameters,
        SchNetParameters, PhysNetParameters, PaiNNParameters,
        TensorNetParameters]
            Parameters for the potential model.
        training_parameter : TrainingParameters
            Parameters for the training process.
        runtime_parameter : RuntimeParameters
            Parameters for runtime configuration.
        dataset_statistic : Dict[str, Dict[str, unit.Quantity]]
            Dataset statistics such as mean and standard deviation.
        use_default_dataset_statistic: bool
            Whether to use default dataset statistic
        optimizer_class : Type[Optimizer], optional
            The optimizer class to use for training, by default
            torch.optim.AdamW.
        potential_seed: Optional[int], optional
            Seed to initialize the potential training adapter, default is None.
        verbose : bool, optional
            If True, enables verbose logging, by default False.
        """

        super().__init__()

        # Assign parameters to instance variables
        self.dataset_parameter = dataset_parameter
        self.potential_parameter = potential_parameter
        self.training_parameter = training_parameter
        self.runtime_parameter = runtime_parameter
        self.verbose = verbose

        # Setup data module
        self.datamodule = self.setup_datamodule()
        # Read and assign provided dataset statistics
        self.dataset_statistic = (
            self.read_dataset_statistics()
            if not use_default_dataset_statistic
            else dataset_statistic
        )
        self.experiment_logger = self.setup_logger()
        self.callbacks = self.setup_callbacks()
        self.trainer = self.setup_trainer()
        self.optimizer_class = optimizer_class
        self.lightning_module = self.setup_lightning_module(potential_seed)

    def read_dataset_statistics(
        self,
    ) -> dict[str, dict[str, Any]]:
        """
        Read and log dataset statistics.

        Returns
        -------
        Dict[str, float]
            The dataset statistics.
        """
        from modelforge.potential.utils import (
            convert_str_to_unit_in_dataset_statistics,
            read_dataset_statistics,
        )

        # read toml file
        dataset_statistic = read_dataset_statistics(
            self.datamodule.dataset_statistic_filename
        )
        # convert dictionary of str:str to str:units
        dataset_statistic = convert_str_to_unit_in_dataset_statistics(dataset_statistic)
        log.info(
            f"Setting per_atom_energy_mean and per_atom_energy_stddev for {self.potential_parameter.potential_name}"
        )
        log.info(
            f"per_atom_energy_mean: {dataset_statistic['training_dataset_statistics']['per_atom_energy_mean']}"
        )
        log.info(
            f"per_atom_energy_stddev: {dataset_statistic['training_dataset_statistics']['per_atom_energy_stddev']}"
        )
        return dataset_statistic

    def setup_datamodule(self) -> DataModule:
        """
        Set up the DataModule for the dataset.

        Returns
        -------
        DataModule
            Configured DataModule instance.
        """
        from modelforge.dataset.dataset import DataModule
        from modelforge.dataset.utils import REGISTERED_SPLITTING_STRATEGIES

        dm = DataModule(
            name=self.dataset_parameter.dataset_name,
            batch_size=self.training_parameter.batch_size,
            remove_self_energies=self.training_parameter.remove_self_energies,
            shift_center_of_mass_to_origin=self.training_parameter.shift_center_of_mass_to_origin,
            version_select=self.dataset_parameter.version_select,
            local_cache_dir=self.runtime_parameter.local_cache_dir,
            splitting_strategy=REGISTERED_SPLITTING_STRATEGIES[
                self.training_parameter.splitting_strategy.name
            ](
                seed=self.training_parameter.splitting_strategy.seed,
                split=self.training_parameter.splitting_strategy.data_split,
            ),
            regenerate_processed_cache=self.dataset_parameter.regenerate_processed_cache,
            properties_of_interest=self.dataset_parameter.properties_of_interest,
            properties_assignment=self.dataset_parameter.properties_assignment.model_dump(),
        )
        dm.prepare_data()
        dm.setup()
        return dm

    def setup_lightning_module(
        self, potential_seed: Optional[int] = None
    ) -> pL.LightningModule:
        """
        Set up the model for training.

        Parameters
        ----------
        potential_seed : int, optional
            Seed to be used to initialize the potential, by default None.

        Returns
        -------
        nn.Module
            Configured model instance, wrapped in a TrainingAdapter.
        """

        # Initialize model
        return TrainingAdapter(
            potential_parameter=self.potential_parameter,
            dataset_statistic=self.dataset_statistic,
            training_parameter=self.training_parameter,
            optimizer_class=self.optimizer_class,
            nr_of_training_batches=len(self.datamodule.train_dataloader()),
            potential_seed=potential_seed,
        )

    def setup_logger(self) -> pL.loggers.Logger:
        """
        Set up the experiment logger based on the configuration.

        Returns
        -------
        pL.loggers.Logger
            Configured logger instance.
        """

        experiment_name = self._format_experiment_name(
            self.runtime_parameter.experiment_name
        )

        if self.training_parameter.experiment_logger.logger_name == "tensorboard":
            from lightning.pytorch.loggers import TensorBoardLogger

            logger = (
                TensorBoardLogger(
                    save_dir=str(
                        self.training_parameter.experiment_logger.tensorboard_configuration.save_dir
                    ),
                    name=experiment_name,
                ),
            )

            log.debug(f'tags: {self._generate_tags(["tensorboard"])}')
        elif self.training_parameter.experiment_logger.logger_name == "wandb":
            from modelforge.utils.io import check_import

            check_import("wandb")
            from lightning.pytorch.loggers import WandbLogger

            logger = WandbLogger(
                save_dir=str(
                    self.training_parameter.experiment_logger.wandb_configuration.save_dir
                ),
                log_model=self.training_parameter.experiment_logger.wandb_configuration.log_model,
                project=self.training_parameter.experiment_logger.wandb_configuration.project,
                group=self.training_parameter.experiment_logger.wandb_configuration.group,
                job_type=self.training_parameter.experiment_logger.wandb_configuration.job_type,
                tags=self._generate_tags(
                    self.training_parameter.experiment_logger.wandb_configuration.tags
                ),
                notes=self.training_parameter.experiment_logger.wandb_configuration.notes,
                name=experiment_name,
            )
        else:
            raise ValueError("Unsupported logger type.")
        return logger

    def setup_callbacks(self) -> List[Any]:
        """
        Set up the callbacks for the trainer.
        The callbacks include early stopping (optional), model checkpointing, and stochastic weight averaging (optional).


        Returns
        -------
        List[Any]
            List of configured callbacks.
        """
        from lightning.pytorch.callbacks import (
            EarlyStopping,
            ModelCheckpoint,
            StochasticWeightAveraging,
            Callback,
        )

        callbacks = []
        if self.training_parameter.stochastic_weight_averaging:
            callbacks.append(
                StochasticWeightAveraging(
                    **self.training_parameter.stochastic_weight_averaging.model_dump()
                )
            )

        if self.training_parameter.early_stopping:
            callbacks.append(
                EarlyStopping(**self.training_parameter.early_stopping.model_dump())
            )

        # Save the best model based on the validation loss
        # NOTE: The filename is formatted as "best_{potential_name}-{dataset_name}-{epoch:02d}"
        checkpoint_filename = (
            f"best_{self.potential_parameter.potential_name}-{self.dataset_parameter.dataset_name}"
            + "-{epoch:02d}"
        )
        callbacks.append(
            ModelCheckpoint(
                save_top_k=2,
                monitor=self.training_parameter.monitor,
                filename=checkpoint_filename,
            )
        )

        # compute gradient norm
        class GradNormCallback(Callback):
            """
            Logs the gradient norm.
            """

            def on_before_optimizer_step(self, trainer, pl_module, optimizer):
                pl_module.log("grad_norm/model", gradient_norm(pl_module))

        if self.training_parameter.log_norm:
            callbacks.append(GradNormCallback())

        return callbacks

    def setup_trainer(self) -> Trainer:
        """
        Set up the Trainer for training.

        Returns
        -------
        Trainer
            Configured Trainer instance.
        """
        from lightning import Trainer

        # if devices is a list (but longer than 1)
        if (
            isinstance(self.runtime_parameter.devices, list)
            and len(self.runtime_parameter.devices) > 1
        ) or (
            isinstance(self.runtime_parameter.devices, int)
            and self.runtime_parameter.devices > 1
        ):
            from lightning.pytorch.strategies import DDPStrategy

            strategy = DDPStrategy(find_unused_parameters=True)
        else:
            strategy = "auto"
        if self.training_parameter.profiler is not None:
            log.debug(f"Using profiler {self.training_parameter.profiler}")

        trainer = Trainer(
            strategy=strategy,
            max_epochs=self.training_parameter.number_of_epochs,
            min_epochs=self.training_parameter.min_number_of_epochs,
            num_nodes=self.runtime_parameter.number_of_nodes,
            devices=self.runtime_parameter.devices,
            accelerator=self.runtime_parameter.accelerator,
            logger=self.experiment_logger,
            callbacks=self.callbacks,
            benchmark=True,
            inference_mode=False,
            limit_train_batches=self.training_parameter.limit_train_batches,
            limit_val_batches=self.training_parameter.limit_val_batches,
            limit_test_batches=self.training_parameter.limit_test_batches,
            profiler=self.training_parameter.profiler,
            num_sanity_val_steps=1,
            gradient_clip_val=5.0,  # FIXME: hardcoded for now
            log_every_n_steps=self.runtime_parameter.log_every_n_steps,
            enable_model_summary=True,
            enable_progress_bar=self.runtime_parameter.verbose,  # if true will show progress bar
        )
        return trainer

    def train_potential(self) -> Trainer:
        """
        Run the training, validation, and testing processes.

        Returns
        -------
        Trainer
            The configured trainer instance after running the training process.
        """
        self.trainer.fit(
            self.lightning_module,
            train_dataloaders=self.datamodule.train_dataloader(
                num_workers=self.dataset_parameter.num_workers,
                pin_memory=self.dataset_parameter.pin_memory,
            ),
            val_dataloaders=self.datamodule.val_dataloader(),
            ckpt_path=(
                self.runtime_parameter.checkpoint_path
                if self.runtime_parameter.checkpoint_path != "None"
                else None
            ),  # NOTE: automatically resumes training from checkpoint
        )

        self.trainer.validate(
            model=self.lightning_module,
            dataloaders=self.datamodule.val_dataloader(),
            ckpt_path="best",
            verbose=True,
        )

        self.trainer.test(
            model=self.lightning_module,
            dataloaders=self.datamodule.test_dataloader(),
            ckpt_path="best",
            verbose=True,
        )
        return self.trainer

    def config_prior(self):
        """
        Configures model-specific priors if the model implements them.
        """
        if hasattr(self.lightning_module, "_config_prior"):
            return self.lightning_module._config_prior()

        log.warning("Model does not implement _config_prior().")
        raise NotImplementedError()

    def _format_experiment_name(self, experiment_name: str) -> str:
        """
        Replace the placeholders in the experiment name with the actual values.

        Parameters
        ----------
        experiment_name : str
            The experiment name with placeholders.

        Returns
        -------
        str
            The experiment name with the placeholders replaced.
        """
        # replace placeholders in the experiment name
        experiment_name = experiment_name.replace(
            "{potential_name}", self.potential_parameter.potential_name
        )
        experiment_name = experiment_name.replace(
            "{dataset_name}", self.dataset_parameter.dataset_name
        )
        return experiment_name

    def _generate_tags(self, tags: List[str]) -> List[str]:
        """Generates tags for the experiment."""
        import modelforge

        try:
            version = modelforge.__version__
        except:
            # for editable local install
            from modelforge._version import __version__

            version = __version__
        losses = [
            f"loss-{loss}"
            for loss in self.training_parameter.loss_parameter.loss_components
        ]
        tags.extend(
            [
                str(version),
                self.dataset_parameter.dataset_name,
                self.potential_parameter.potential_name,
                # f"loss-{'-'.join(self.training_parameter.loss_parameter.loss_components)}",
            ]
        )
        tags.extend(losses)

        return tags


from typing import List, Optional, Union


def read_config(
    condensed_config_path: Optional[str] = None,
    training_parameter_path: Optional[str] = None,
    dataset_parameter_path: Optional[str] = None,
    potential_parameter_path: Optional[str] = None,
    runtime_parameter_path: Optional[str] = None,
    accelerator: Optional[str] = None,
    devices: Optional[Union[int, List[int]]] = None,
    number_of_nodes: Optional[int] = None,
    experiment_name: Optional[str] = None,
    save_dir: Optional[str] = None,
    local_cache_dir: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    log_every_n_steps: Optional[int] = None,
    simulation_environment: Optional[str] = None,
):
    """
    Reads one or more TOML configuration files and loads them into the pydantic
    models.

    Parameters
    ----------
    condensed_config_path : Optional[str], optional
        Path to the TOML configuration that contains all parameters for the
        dataset, potential, training, and runtime parameters. Any other provided
        configuration files will be ignored.
    training_parameter_path : Optional[str], optional
        Path to the TOML file defining the training parameters.
    dataset_parameter_path : Optional[str], optional
        Path to the TOML file defining the dataset parameters.
    potential_parameter_path : Optional[str], optional
        Path to the TOML file defining the potential parameters.
    runtime_parameter_path : Optional[str], optional
        Path to the TOML file defining the runtime parameters. If this is not
        provided, the code will attempt to use the runtime parameters provided
        as arguments.
    accelerator : Optional[str], optional
        Accelerator type to use. If provided, this overrides the accelerator
        type in the runtime_defaults configuration.
    devices : Optional[Union[int, List[int]]], optional
        Device index/indices to use. If provided, this overrides the devices in
        the runtime_defaults configuration.
    number_of_nodes : Optional[int], optional
        Number of nodes to use. If provided, this overrides the number of nodes
        in the runtime_defaults configuration.
    experiment_name : Optional[str], optional
        Name of the experiment. If provided, this overrides the experiment name
        in the runtime_defaults configuration.
    save_dir : Optional[str], optional
        Directory to save the model. If provided, this overrides the save
        directory in the runtime_defaults configuration.
    local_cache_dir : Optional[str], optional
        Local cache directory. If provided, this overrides the local cache
        directory in the runtime_defaults configuration.
    checkpoint_path : Optional[str], optional
        Path to the checkpoint file. If provided, this overrides the checkpoint
        path in the runtime_defaults configuration.
    log_every_n_steps : Optional[int], optional
        Number of steps to log. If provided, this overrides the
        log_every_n_steps in the runtime_defaults configuration.
    simulation_environment : Optional[str], optional
        Simulation environment. If provided, this overrides the simulation
        environment in the runtime_defaults configuration.

    Returns
    -------
    Tuple[TrainingParameters, DatasetParameters, T_NNP_Parameters,
    RuntimeParameters]
        Tuple containing the training, dataset, potential, and runtime
        parameters.
    """
    import toml

    # Initialize the config dictionaries
    training_config_dict = {}
    dataset_config_dict = {}
    potential_config_dict = {}
    runtime_config_dict = {}

    if condensed_config_path is not None:
        # Load all configurations from a single condensed TOML file
        config = toml.load(condensed_config_path)
        log.info(f"Reading config from : {condensed_config_path}")

        training_config_dict = config.get("training", {})
        dataset_config_dict = config.get("dataset", {})
        potential_config_dict = config.get("potential", {})
        runtime_config_dict = config.get("runtime", {})

    else:
        if training_parameter_path:
            training_config_dict = toml.load(training_parameter_path).get(
                "training", {}
            )
        if dataset_parameter_path:
            dataset_config_dict = toml.load(dataset_parameter_path).get("dataset", {})
        if potential_parameter_path:
            potential_config_dict = toml.load(potential_parameter_path).get(
                "potential", {}
            )
        if runtime_parameter_path:
            runtime_config_dict = toml.load(runtime_parameter_path).get("runtime", {})

    # Override runtime configuration with command-line arguments if provided
    runtime_overrides = {
        "accelerator": accelerator,
        "devices": devices,
        "number_of_nodes": number_of_nodes,
        "experiment_name": experiment_name,
        "save_dir": save_dir,
        "local_cache_dir": local_cache_dir,
        "checkpoint_path": checkpoint_path,
        "log_every_n_steps": log_every_n_steps,
        "simulation_environment": simulation_environment,
    }

    for key, value in runtime_overrides.items():
        if value is not None:
            runtime_config_dict[key] = value

    # Load and instantiate the data classes with the merged configuration
    from modelforge.dataset.dataset import DatasetParameters
    from modelforge.potential import _Implemented_NNP_Parameters
    from modelforge.train.parameters import RuntimeParameters, TrainingParameters

    potential_name = potential_config_dict["potential_name"]
    PotentialParameters = (
        _Implemented_NNP_Parameters.get_neural_network_parameter_class(potential_name)
    )

    dataset_parameters = DatasetParameters(**dataset_config_dict)
    training_parameters = TrainingParameters(**training_config_dict)
    runtime_parameters = RuntimeParameters(**runtime_config_dict)
    potential_parameter = PotentialParameters(**potential_config_dict)

    return (
        training_parameters,
        dataset_parameters,
        potential_parameter,
        runtime_parameters,
    )


def read_config_and_train(
    condensed_config_path: Optional[str] = None,
    training_parameter_path: Optional[str] = None,
    dataset_parameter_path: Optional[str] = None,
    potential_parameter_path: Optional[str] = None,
    runtime_parameter_path: Optional[str] = None,
    accelerator: Optional[str] = None,
    devices: Optional[Union[int, List[int]]] = None,
    number_of_nodes: Optional[int] = None,
    experiment_name: Optional[str] = None,
    save_dir: Optional[str] = None,
    local_cache_dir: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    log_every_n_steps: Optional[int] = None,
    simulation_environment: Optional[str] = "PyTorch",
):
    """
    Reads one or more TOML configuration files and performs training based on the parameters.

    Parameters
    ----------
    condensed_config_path : str, optional
        Path to the TOML configuration that contains all parameters for the dataset, potential, training, and runtime parameters.
        Any other provided configuration files will be ignored.
    training_parameter_path : str, optional
        Path to the TOML file defining the training parameters.
    dataset_parameter_path : str, optional
        Path to the TOML file defining the dataset parameters.
    potential_parameter_path : str, optional
        Path to the TOML file defining the potential parameters.
    runtime_parameter_path : str, optional
        Path to the TOML file defining the runtime parameters. If this is not provided, the code will attempt to use
        the runtime parameters provided as arguments.
    accelerator : str, optional
        Accelerator type to use.  If provided, this  overrides the accelerator type in the runtime_defaults configuration.
    devices : int|List[int], optional
        Device index/indices to use.  If provided, this overrides the devices in the runtime_defaults configuration.
    number_of_nodes : int, optional
        Number of nodes to use.  If provided, this overrides the number of nodes in the runtime_defaults configuration.
    experiment_name : str, optional
        Name of the experiment.  If provided, this overrides the experiment name in the runtime_defaults configuration.
    save_dir : str, optional
        Directory to save the model.  If provided, this overrides the save directory in the runtime_defaults configuration.
    local_cache_dir : str, optional
        Local cache directory.  If provided, this overrides the local cache directory in the runtime_defaults configuration.
    checkpoint_path : str, optional
        Path to the checkpoint file.  If provided, this overrides the checkpoint path in the runtime_defaults configuration.
    log_every_n_steps : int, optional
        Number of steps to log.  If provided, this overrides the log_every_n_steps in the runtime_defaults configuration.
    simulation_environment : str, optional
        Simulation environment.  If provided, this overrides the simulation environment in the runtime_defaults configuration.

    Returns
    -------
    Trainer
        The configured trainer instance after running the training process.
    """
    from modelforge.potential.potential import NeuralNetworkPotentialFactory

    (
        training_parameter,
        dataset_parameter,
        potential_parameter,
        runtime_parameter,
    ) = read_config(
        condensed_config_path=condensed_config_path,
        training_parameter_path=training_parameter_path,
        dataset_parameter_path=dataset_parameter_path,
        potential_parameter_path=potential_parameter_path,
        runtime_parameter_path=runtime_parameter_path,
        accelerator=accelerator,
        devices=devices,
        number_of_nodes=number_of_nodes,
        experiment_name=experiment_name,
        save_dir=save_dir,
        local_cache_dir=local_cache_dir,
        checkpoint_path=checkpoint_path,
        log_every_n_steps=log_every_n_steps,
        simulation_environment=simulation_environment,
    )

    trainer = NeuralNetworkPotentialFactory.generate_trainer(
        potential_parameter=potential_parameter,
        training_parameter=training_parameter,
        dataset_parameter=dataset_parameter,
        runtime_parameter=runtime_parameter,
    )

    return trainer.train_potential()
