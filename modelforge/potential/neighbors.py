"""
This file contains classes for computing pairs and neighbors.
"""

import torch
from loguru import logger as log
from modelforge.dataset.dataset import NNPInput

from typing import Union, NamedTuple


class PairlistData(NamedTuple):
    """
    A namedtuple to store the outputs of the Pairlist and Neighborlist forward methods.

    Attributes
    ----------
    pair_indices : torch.Tensor
        A tensor of shape (2, n_pairs) containing the indices of the interacting atom pairs.
    d_ij : torch.Tensor
        A tensor of shape (n_pairs, 1) containing the Euclidean distances between the atoms in each pair.
    r_ij : torch.Tensor
        A tensor of shape (n_pairs, 3) containing the displacement vectors between the atoms in each pair.
    """

    pair_indices: torch.Tensor
    d_ij: torch.Tensor
    r_ij: torch.Tensor


class Pairlist(torch.nn.Module):
    def __init__(self, only_unique_pairs: bool = False):
        """
         Handle pair list calculations for systems, returning indices, distances
         and distance vectors for atom pairs within a certain cutoff.

        Parameters
        ----------
        only_unique_pairs : bool, optional
            If True, only unique pairs are returned (default is False).
        """
        super().__init__()
        self.only_unique_pairs = only_unique_pairs

    def enumerate_all_pairs(self, atomic_subsystem_indices: torch.Tensor):
        """
        Compute all pairs of atoms and their distances.

        Parameters
        ----------
        atomic_subsystem_indices : torch.Tensor
            Atom indices to indicate which atoms belong to which molecule.
            Note in all cases, the values in this tensor must be numbered from 0 to n_molecules - 1 sequentially, with no gaps in the numbering. E.g., [0,0,0,1,1,2,2,2 ...].
            This is the case for all internal data structures, and thus no validation is performed in this routine. If the data is not structured in this way, the results will be incorrect.

        Returns
        -------
        torch.Tensor
            Pair indices for all atom pairs.
        """

        # get device that passed tensors lives on, initialize on the same device
        device = atomic_subsystem_indices.device

        # if there is only one molecule, we do not need to use additional looping and offsets
        if torch.sum(atomic_subsystem_indices) == 0:
            n = len(atomic_subsystem_indices)
            if self.only_unique_pairs:
                i_final_pairs, j_final_pairs = torch.triu_indices(
                    n, n, 1, device=device
                )
            else:
                # Repeat each number n-1 times for i_indices
                i_final_pairs = torch.repeat_interleave(
                    torch.arange(n, device=device),
                    repeats=n - 1,
                )

                # Correctly construct j_indices
                j_final_pairs = torch.cat(
                    [
                        torch.cat(
                            (
                                torch.arange(i, device=device),
                                torch.arange(i + 1, n, device=device),
                            )
                        )
                        for i in range(n)
                    ]
                )

        else:
            # if we have more than one molecule, we will take into account molecule size and offsets when
            # calculating pairs, as using the approach above is not memory efficient for datasets with large molecules
            # and/or larger batch sizes; while not likely a problem on higher end GPUs with large amounts of memory
            # cheaper commodity and mobile GPUs may have issues

            # atomic_subsystem_indices are always numbered from 0 to n_molecules
            # - 1 e.g., a single molecule will be [0, 0, 0, 0 ... ] and a batch
            # of molecules will always start at 0 and increment [ 0, 0, 0, 1, 1,
            # 1, ...] As such, we can use bincount, as there are no gaps in the
            # numbering

            # Note if the indices are not numbered from 0 to n_molecules - 1, this will not work
            # E.g., bincount on [3,3,3, 4,4,4, 5,5,5] will return [0,0,0,3,3,3,3,3,3]
            # as we have no values for 0, 1, 2
            # using a combination of unique and argsort would make this work for any numbering ordering
            # but that is not how the data ends up being structured internally, and thus is not needed
            repeats = torch.bincount(atomic_subsystem_indices)
            offsets = torch.cat(
                (torch.tensor([0], device=device), torch.cumsum(repeats, dim=0)[:-1])
            )

            i_indices = torch.cat(
                [
                    torch.repeat_interleave(
                        torch.arange(o, o + r, device=device), repeats=r
                    )
                    for r, o in zip(repeats, offsets)
                ]
            )
            j_indices = torch.cat(
                [
                    torch.cat([torch.arange(o, o + r, device=device) for _ in range(r)])
                    for r, o in zip(repeats, offsets)
                ]
            )

            if self.only_unique_pairs:
                # filter out pairs that are not unique
                unique_pairs_mask = i_indices < j_indices
                i_final_pairs = i_indices[unique_pairs_mask]
                j_final_pairs = j_indices[unique_pairs_mask]
            else:
                # filter out identical values
                unique_pairs_mask = i_indices != j_indices
                i_final_pairs = i_indices[unique_pairs_mask]
                j_final_pairs = j_indices[unique_pairs_mask]

        # concatenate to form final (2, n_pairs) tensor
        pair_indices = torch.stack((i_final_pairs, j_final_pairs))

        return pair_indices.to(device)

    def construct_initial_pairlist_using_numpy(
        self, atomic_subsystem_indices: torch.Tensor
    ):
        """Compute all pairs of atoms and also return counts of the number of pairs for each molecule in batch.

        Parameters
        ----------
        atomic_subsystem_indices : torch.Tensor
            Atom indices to indicate which atoms belong to which molecule.

        Returns
        -------
        pair_indices : np.ndarray, shape (2, n_pairs)
            Pairs of atom indices, 0-indexed for each molecule
        number_of_pairs : np.ndarray, shape (n_molecules)
            The number to index into pair_indices for each molecule

        """

        # atomic_subsystem_indices are always numbered from 0 to n_molecules - 1
        # e.g., a single molecule will be [0, 0, 0, 0 ... ]
        # and a batch of molecules will always start at 0 and increment [ 0, 0, 0, 1, 1, 1, ...]
        # As such, we can use bincount, as there are no gaps in the numbering
        # Note if the indices are not numbered from 0 to n_molecules - 1, this will not work
        # E.g., bincount on [3,3,3, 4,4,4, 5,5,5] will return [0,0,0,3,3,3,3,3,3]
        # as we have no values for 0, 1, 2
        # using a combination of unique and argsort would make this work for any numbering ordering
        # but that is not how the data ends up being structured internally, and thus is not needed

        import numpy as np

        # get the number of atoms in each molecule
        repeats = np.bincount(atomic_subsystem_indices)

        # calculate the number of pairs for each molecule, using simple permutation
        npairs_by_molecule = np.array([r * (r - 1) for r in repeats], dtype=np.int16)

        i_indices = np.concatenate(
            [
                np.repeat(
                    np.arange(
                        0,
                        r,
                        dtype=np.int16,
                    ),
                    repeats=r,
                )
                for r in repeats
            ]
        )
        j_indices = np.concatenate(
            [
                np.concatenate([np.arange(0, 0 + r, dtype=np.int16) for _ in range(r)])
                for r in repeats
            ]
        )

        # filter out identical pairs where i==j
        unique_pairs_mask = i_indices != j_indices
        i_final_pairs = i_indices[unique_pairs_mask]
        j_final_pairs = j_indices[unique_pairs_mask]

        # concatenate to form final (2, n_pairs) vector
        pair_indices = np.stack((i_final_pairs, j_final_pairs))

        return pair_indices, npairs_by_molecule

    def calculate_r_ij(
        self, pair_indices: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Compute displacement vectors between atom pairs.

        Parameters
        ----------
        pair_indices : torch.Tensor
            Atom indices for pairs of atoms. Shape: [2, n_pairs].
        positions : torch.Tensor
            Atom positions. Shape: [atoms, 3].

        Returns
        -------
        torch.Tensor
            Displacement vectors between atom pairs. Shape: [n_pairs, 3].
        """
        # Select the pairs of atom coordinates from the positions
        selected_positions = positions.index_select(0, pair_indices.view(-1)).view(
            2, -1, 3
        )
        return selected_positions[1] - selected_positions[0]

    def calculate_d_ij(self, r_ij: torch.Tensor) -> torch.Tensor:
        """
        ompute Euclidean distances between atoms in each pair.

        Parameters
        ----------
        r_ij : torch.Tensor
            Displacement vectors between atoms in a pair. Shape: [n_pairs, 3].

        Returns
        -------
        torch.Tensor
            Euclidean distances. Shape: [n_pairs, 1].
        """
        return r_ij.norm(dim=1).unsqueeze(1)

    def forward(
        self,
        positions: torch.Tensor,
        atomic_subsystem_indices: torch.Tensor,
    ) -> PairlistData:
        """
        Performs the forward pass of the Pairlist module.

        Parameters
        ----------
        positions : torch.Tensor
            Atom positions. Shape: [nr_atoms, 3].
        atomic_subsystem_indices (torch.Tensor, shape (nr_atoms_per_systems)):
            Atom indices to indicate which atoms belong to which molecule.

        Returns
        -------
        PairListOutputs: A dataclass containing the following attributes:
            pair_indices (torch.Tensor): A tensor of shape (2, n_pairs) containing the indices of the interacting atom pairs.
            d_ij (torch.Tensor): A tensor of shape (n_pairs, 1) containing the Euclidean distances between the atoms in each pair.
            r_ij (torch.Tensor): A tensor of shape (n_pairs, 3) containing the displacement vectors between the atoms in each pair.
        """
        pair_indices = self.enumerate_all_pairs(
            atomic_subsystem_indices,
        )
        r_ij = self.calculate_r_ij(pair_indices, positions)
        return PairlistData(
            pair_indices=pair_indices,
            d_ij=self.calculate_d_ij(r_ij),
            r_ij=r_ij,
        )


class OrthogonalDisplacementFunction(torch.nn.Module):
    def __init__(self):
        """
        Compute displacement vectors between pairs of atoms, considering periodic boundary conditions if used.

        """
        super().__init__()

    def forward(
        self,
        coordinate_i: torch.Tensor,
        coordinate_j: torch.Tensor,
        box_vectors: torch.Tensor,
        is_periodic: torch.Tensor,
    ):
        """
        Compute displacement vectors and Euclidean distances between atom pairs.

        Parameters
        ----------
        coordinate_i : torch.Tensor
            Coordinates of the first atom in each pair. Shape: [n_pairs, 3].
        coordinate_j : torch.Tensor
            Coordinates of the second atom in each pair. Shape: [n_pairs, 3].
        box_vectors : torch.Tensor
            Box vectors defining the periodic boundary conditions. Shape: [3, 3].
        is_periodic : bool
            Whether to apply periodic boundary conditions.
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Displacement vectors (r_ij) of shape [n_pairs, 3] and distances (d_ij) of shape [n_pairs, 1].
        """
        r_ij = coordinate_i - coordinate_j

        if is_periodic == True:
            # Note, since box length may change, we need to update each time if periodic
            # reinitializing this vector each time does not have a significant performance impact

            box_lengths = torch.zeros(
                3, device=box_vectors.device, dtype=box_vectors.dtype
            )

            box_lengths[0] = box_vectors[0][0]
            box_lengths[1] = box_vectors[1][1]
            box_lengths[2] = box_vectors[2][2]

            r_ij = (
                torch.remainder(r_ij + box_lengths / 2, box_lengths) - box_lengths / 2
            )

        d_ij = torch.norm(r_ij, dim=1, keepdim=True, p=2)
        return r_ij, d_ij


class NeighborlistBruteNsq(torch.nn.Module):
    """
    Brute force N^2 neighbor list calculation for inference implemented fully in PyTorch.

    This is compatible with TorchScript.


    """

    def __init__(
        self,
        cutoff: float,
        displacement_function: OrthogonalDisplacementFunction,
        only_unique_pairs: bool = False,
    ):
        """
        Compute neighbor lists for inference, filtering pairs based on a cutoff distance.

        Parameters
        ----------
        cutoff : float
            The cutoff distance for neighbor list calculations.
        displacement_function : OrthogonalDisplacementFunction
            The function to calculate displacement vectors and distances between atom pairs, taking into account
            the specified boundary conditions.
        only_unique_pairs : bool, optional
            Whether to only use unique pairs in the pair list calculation, by default False.
        """

        super().__init__()

        self.register_buffer("cutoff", torch.tensor(cutoff))
        self.register_buffer("only_unique_pairs", torch.tensor(only_unique_pairs))
        self.displacement_function = displacement_function

        self.indices = torch.tensor([])
        self.i_final_pairs = torch.tensor([])
        self.j_final_pairs = torch.tensor([])
        log.info("Initializing Brute Force N^2 Neighborlist")

    def _copy_to_nonunique(
        self,
        i_pairs: torch.Tensor,
        j_pairs: torch.Tensor,
        d_ij: torch.Tensor,
        r_ij: torch.Tensor,
        total_unique_pairs: int,
    ):
        r_ij_full = torch.zeros(
            total_unique_pairs * 2, 3, dtype=r_ij.dtype, device=r_ij.device
        )
        d_ij_full = torch.zeros(
            total_unique_pairs * 2, 1, dtype=d_ij.dtype, device=d_ij.device
        )

        r_ij_full[0:total_unique_pairs] = r_ij
        r_ij_full[total_unique_pairs : 2 * total_unique_pairs] = -r_ij

        d_ij_full[0:total_unique_pairs] = d_ij
        d_ij_full[total_unique_pairs : 2 * total_unique_pairs] = d_ij

        pairs_full = torch.zeros(
            2, total_unique_pairs * 2, dtype=torch.int64, device=i_pairs.device
        )

        pairs_full[0][0:total_unique_pairs] = i_pairs
        pairs_full[1][0:total_unique_pairs] = j_pairs
        pairs_full[0][total_unique_pairs : 2 * total_unique_pairs] = j_pairs
        pairs_full[1][total_unique_pairs : 2 * total_unique_pairs] = i_pairs

        return pairs_full, d_ij_full, r_ij_full

    def forward(self, data: NNPInput):
        """
        Prepares the input tensors for passing to the model.

        This method handles general input manipulation, such as calculating
        distances and generating the pair list. It also calls the model-specific
        input preparation.

        Parameters
        ----------
        data : NNPInput
            The input data provided by the dataset, containing atomic numbers,
            positions, and other necessary information.

        Returns
        -------
        PairListOutputs
            Contains pair indices, distances (d_ij), and displacement vectors (r_ij) for atom pairs within the cutoff.
        """
        # ---------------------------
        # general input manipulation
        positions = data.positions
        atomic_subsystem_indices = data.atomic_subsystem_indices

        n = atomic_subsystem_indices.size(0)

        # avoid reinitializing indices if they are already set and haven't changed
        if self.indices.shape[0] != n:
            # Generate a range of indices from 0 to n-1
            self.indices = torch.arange(n, device=atomic_subsystem_indices.device)

            # Create a meshgrid of indices
            self.i_final_pairs, self.j_final_pairs = torch.meshgrid(
                self.indices, self.indices, indexing="ij"
            )
            # We will only consider unique pairs; for non-unique pairs we can just appropriately copy
            # the data as it will be faster than extra computations.
            mask = self.i_final_pairs < self.j_final_pairs

            self.i_final_pairs = self.i_final_pairs[mask]
            self.j_final_pairs = self.j_final_pairs[mask]

        # calculate r_ij and d_ij
        r_ij, d_ij = self.displacement_function(
            positions[self.i_final_pairs],
            positions[self.j_final_pairs],
            data.box_vectors,
            data.is_periodic,
        )
        in_cutoff = (d_ij <= self.cutoff).squeeze()
        total_pairs = in_cutoff.sum()

        if self.only_unique_pairs:
            # using this instead of torch.stack to ensure that if we only have a single pair
            # we don't run into an issue with tensor shapes.
            # note this will fail if there are no interacting pairs

            pairs = torch.zeros(
                2, total_pairs, dtype=torch.int64, device=positions.device
            )

            pairs[0] = self.i_final_pairs[in_cutoff]
            pairs[1] = self.j_final_pairs[in_cutoff]

            return PairlistData(
                pair_indices=pairs,
                d_ij=d_ij[in_cutoff],
                r_ij=r_ij[in_cutoff],
            )

        else:
            pairs_full, d_ij_full, r_ij_full = self._copy_to_nonunique(
                self.i_final_pairs[in_cutoff],
                self.j_final_pairs[in_cutoff],
                d_ij[in_cutoff],
                r_ij[in_cutoff],
                total_pairs,
            )
            return PairlistData(
                pair_indices=pairs_full,
                d_ij=d_ij_full,
                r_ij=r_ij_full,
            )


class NeighborlistVerletNsq(torch.nn.Module):
    """
    Verlet neighbor list calculation for inference implemented fully in PyTorch.

    Rebuilding of the neighborlist uses an N^2 approach.  Rebuilding occurs when
    the maximum displacement of any particle exceeds half the skin distance.

    """

    def __init__(
        self,
        cutoff: float,
        skin: float,
        displacement_function: OrthogonalDisplacementFunction,
        only_unique_pairs: bool = False,
    ):
        """
        Compute neighbor lists for inference, filtering pairs based on a cutoff distance.

        Parameters
        ----------
        cutoff : float
            The cutoff distance for neighbor list calculations.
        skin : float
            The skin distance for neighbor list calculations.
        displacement_function : OrthogonalDisplacementFunction
            The function to calculate displacement vectors and distances between atom pairs, taking into account
            the specified boundary conditions.
        only_unique_pairs : bool, optional
            Whether to only use unique pairs in the pair list calculation, by
            default True. This should be set to True for all message passing
            networks.
        """

        super().__init__()

        self.register_buffer("cutoff", torch.tensor(cutoff))
        self.skin = skin
        self.half_skin = skin * 0.5
        self.cutoff_plus_skin = cutoff + skin
        self.only_unique_pairs = only_unique_pairs

        self.displacement_function = displacement_function
        self.indices = torch.tensor([])
        self.i_pairs = torch.tensor([])
        self.j_pairs = torch.tensor([])

        self.positions_old = torch.tensor([])
        self.nlist_pairs = torch.tensor([])
        self.builds = 0
        self.box_vectors = torch.zeros([3, 3])

        log.info("Initializing Verlet Neighborlist with N^2 building routine.")

    def _check_nlist(
        self, positions: torch.Tensor, box_vectors: torch.Tensor, is_periodic
    ):
        r_ij, d_ij = self.displacement_function(
            self.positions_old, positions, box_vectors, is_periodic
        )

        if torch.any(d_ij > self.half_skin):
            return True
        else:
            return False

    def _init_pairs(self, n_particles: int, device: torch.device):
        self.indices = torch.arange(n_particles, device=device)

        i_pairs, j_pairs = torch.meshgrid(
            self.indices,
            self.indices,
            indexing="ij",
        )

        mask = i_pairs < j_pairs
        self.i_pairs = i_pairs[mask]
        self.j_pairs = j_pairs[mask]

    def _build_nlist(
        self, positions: torch.Tensor, box_vectors: torch.Tensor, is_periodic
    ):
        r_ij, d_ij = self.displacement_function(
            positions[self.i_pairs], positions[self.j_pairs], box_vectors, is_periodic
        )

        in_cutoff = (d_ij < self.cutoff_plus_skin).squeeze()
        self.nlist_pairs = torch.stack(
            [self.i_pairs[in_cutoff], self.j_pairs[in_cutoff]]
        )
        self.builds += 1
        return r_ij[in_cutoff], d_ij[in_cutoff]

    def _copy_to_nonunique(
        self,
        pairs: torch.Tensor,
        d_ij: torch.Tensor,
        r_ij: torch.Tensor,
        total_unique_pairs: int,
    ):
        # this will allow us to copy the data for unique pairs to create the non-unique pairs data
        r_ij_full = torch.zeros(
            total_unique_pairs * 2, 3, dtype=r_ij.dtype, device=r_ij.device
        )
        d_ij_full = torch.zeros(
            total_unique_pairs * 2, 1, dtype=d_ij.dtype, device=d_ij.device
        )

        r_ij_full[0:total_unique_pairs] = r_ij

        # since we are swapping the order of the pairs, the sign changes
        r_ij_full[total_unique_pairs : 2 * total_unique_pairs] = -r_ij

        d_ij_full[0:total_unique_pairs] = d_ij
        d_ij_full[total_unique_pairs : 2 * total_unique_pairs] = d_ij

        pairs_full = torch.zeros(
            2, total_unique_pairs * 2, dtype=torch.int64, device=pairs.device
        )

        pairs_full[0][0:total_unique_pairs] = pairs[0]
        pairs_full[1][0:total_unique_pairs] = pairs[1]
        pairs_full[0][total_unique_pairs : 2 * total_unique_pairs] = pairs[1]
        pairs_full[1][total_unique_pairs : 2 * total_unique_pairs] = pairs[0]

        return pairs_full, d_ij_full, r_ij_full

    def forward(self, data: NNPInput):
        """
        Prepares the input tensors for passing to the model.

        This method handles general input manipulation, such as calculating
        distances and generating the pair list. It also calls the model-specific
        input preparation.

        Parameters
        ----------
        data : NNPInput
            The input data provided by the dataset, containing atomic numbers,
            positions, and other necessary information.

        Returns
        -------
        PairListOutputs
            Contains pair indices, distances (d_ij), and displacement vectors (r_ij) for atom pairs within the cutoff.
        """
        # ---------------------------
        # general input manipulation
        positions = data.positions
        atomic_subsystem_indices = data.atomic_subsystem_indices

        n = atomic_subsystem_indices.size(0)
        # if the initial build we haven't yet set box vectors so set them
        # this is necessary because we need to store them to know if we need to force a rebuild
        # because the box vectors have changed
        if self.builds == 0:
            self.box_vectors = data.box_vectors

        box_changed = torch.any(self.box_vectors != data.box_vectors)

        # avoid reinitializing indices if they are already set and haven't changed
        if self.indices.shape[0] != n:
            self.box_vectors = data.box_vectors
            self.positions_old = positions
            self._init_pairs(n, positions.device)
            r_ij, d_ij = self._build_nlist(
                positions, data.box_vectors, data.is_periodic
            )
        elif box_changed:
            # if the box vectors have changed, we need to rebuild the nlist
            # but do not need to regenerate the pairs
            self.box_vectors = data.box_vectors
            self.positions_old = positions
            r_ij, d_ij = self._build_nlist(
                positions, data.box_vectors, data.is_periodic
            )
        elif self._check_nlist(positions, data.box_vectors, data.is_periodic):
            self.positions_old = positions
            r_ij, d_ij = self._build_nlist(
                positions, data.box_vectors, data.is_periodic
            )
        else:
            r_ij, d_ij = self.displacement_function(
                positions[self.nlist_pairs[0]],
                positions[self.nlist_pairs[1]],
                data.box_vectors,
                data.is_periodic,
            )

        in_cutoff = (d_ij <= self.cutoff).squeeze()
        total_pairs = in_cutoff.sum()

        if self.only_unique_pairs:
            # using this instead of torch.stack to ensure that if we only have a single pair
            # we don't run into an issue with shapes.

            pairs = torch.zeros(
                2, total_pairs, dtype=torch.int64, device=positions.device
            )

            pairs[0] = self.nlist_pairs[0][in_cutoff]
            pairs[1] = self.nlist_pairs[1][in_cutoff]

            return PairlistData(
                pair_indices=pairs,
                d_ij=d_ij[in_cutoff],
                r_ij=r_ij[in_cutoff],
            )

        else:
            pairs_full, d_ij_full, r_ij_full = self._copy_to_nonunique(
                self.nlist_pairs[:, in_cutoff],
                d_ij[in_cutoff],
                r_ij[in_cutoff],
                total_pairs,
            )
            return PairlistData(
                pair_indices=pairs_full,
                d_ij=d_ij_full,
                r_ij=r_ij_full,
            )


class NeighborlistForInference(torch.nn.Module):
    """
    Neighbor list calculation for inference implemented fully in PyTorch, allowing brute and Verlet nsq methods.

    Rebuilding of the neighborlist uses an N^2 approach.  Rebuilding occurs when
    the maximum displacement of any particle exceeds half the skin distance for Verlet lists.

    """

    def __init__(
        self,
        cutoff: float,
        displacement_function: OrthogonalDisplacementFunction,
        only_unique_pairs: bool = False,
    ):
        """
        Compute neighbor lists for inference, filtering pairs based on a cutoff distance.

        Parameters
        ----------
        cutoff : float
            The cutoff distance for neighbor list calculations.
        displacement_function : OrthogonalDisplacementFunction
            The function to calculate displacement vectors and distances between atom pairs, taking into account
            the specified boundary conditions.
        only_unique_pairs : bool, optional
            Whether to only use unique pairs in the pair list calculation, by
            default True. This should be set to True for all message passing
            networks.
        """

        super().__init__()

        self.register_buffer("cutoff", torch.tensor(cutoff))
        self.register_buffer("only_unique_pairs", torch.tensor(only_unique_pairs))
        # set a default value; we can update this when we set the strategy
        self.skin = 0.1

        self.half_skin = self.skin * 0.5
        self.cutoff_plus_skin = self.cutoff + self.skin
        # self.only_unique_pairs = only_unique_pairs

        self.displacement_function = displacement_function
        self.indices = torch.tensor([])
        self.i_pairs = torch.tensor([])
        self.j_pairs = torch.tensor([])

        self.indices = torch.tensor([])

        self.positions_old = torch.tensor([])
        self.nlist_i_pairs = torch.tensor([])
        self.nlist_j_pairs = torch.tensor([])
        self.builds = 0
        self.box_vectors = torch.zeros([3, 3])

        log.info("Initializing Verlet Neighborlist with N^2 building routine.")

    def _check_verlet_nlist(
        self, positions: torch.Tensor, box_vectors: torch.Tensor, is_periodic
    ):
        r_ij, d_ij = self.displacement_function(
            self.positions_old, positions, box_vectors, is_periodic
        )

        if torch.any(d_ij > self.half_skin):
            return True
        else:
            return False

    def _init_verlet_pairs(self, n_particles: int, device: torch.device):
        self.indices = torch.arange(n_particles, device=device)

        i_pairs, j_pairs = torch.meshgrid(
            self.indices,
            self.indices,
            indexing="ij",
        )

        mask = i_pairs < j_pairs
        # self.i_pairs = i_pairs[mask]
        # self.j_pairs = j_pairs[mask]

        return i_pairs[mask], j_pairs[mask]

    def _build_verlet_nlist(
        self, positions: torch.Tensor, box_vectors: torch.Tensor, is_periodic
    ):
        r_ij, d_ij = self.displacement_function(
            positions[self.i_pairs], positions[self.j_pairs], box_vectors, is_periodic
        )

        in_cutoff = (d_ij < self.cutoff_plus_skin).squeeze()
        # self.nlist_i_pairs = self.i_pairs[in_cutoff]
        # self.nlist_j_pairs = self.j_pairs[in_cutoff]

        self.builds += 1
        return (
            self.i_pairs[in_cutoff],
            self.j_pairs[in_cutoff],
            r_ij[in_cutoff],
            d_ij[in_cutoff],
        )

    def _copy_to_nonunique(
        self,
        i_pairs: torch.Tensor,
        j_pairs: torch.Tensor,
        d_ij: torch.Tensor,
        r_ij: torch.Tensor,
        total_unique_pairs: int,
    ):
        # this will allow us to copy the data for unique pairs to create the non-unique pairs data
        r_ij_full = torch.zeros(
            total_unique_pairs * 2, 3, dtype=r_ij.dtype, device=r_ij.device
        )
        d_ij_full = torch.zeros(
            total_unique_pairs * 2, 1, dtype=d_ij.dtype, device=d_ij.device
        )

        r_ij_full[0:total_unique_pairs] = r_ij

        # since we are swapping the order of the pairs, the sign changes
        r_ij_full[total_unique_pairs : 2 * total_unique_pairs] = -r_ij

        d_ij_full[0:total_unique_pairs] = d_ij
        d_ij_full[total_unique_pairs : 2 * total_unique_pairs] = d_ij

        pairs_full = torch.zeros(
            2, total_unique_pairs * 2, dtype=torch.int64, device=i_pairs.device
        )

        pairs_full[0][0:total_unique_pairs] = i_pairs
        pairs_full[1][0:total_unique_pairs] = j_pairs
        pairs_full[0][total_unique_pairs : 2 * total_unique_pairs] = j_pairs
        pairs_full[1][total_unique_pairs : 2 * total_unique_pairs] = i_pairs

        return pairs_full, d_ij_full, r_ij_full

    def set_strategy(self, strategy: str, skin: float = 0.1):
        """
        Set the strategy for rebuilding the neighbor list.

        Parameters
        ----------
        strategy : str
            The strategy to use for rebuilding the neighbor list. Options are "verlet_nsq" or "brute_nsq".
        skin : float, optional
            The skin distance for the Verlet list, by default 0.1.
        """
        self.skin = skin
        self.half_skin = self.skin * 0.5
        self.cutoff_plus_skin = self.cutoff + self.skin

        if strategy == "verlet_nsq":
            self.forward = self._forward_verlet
        elif strategy == "brute_nsq":
            self.forward = self._forward_brute
        else:
            raise ValueError(f"Unknown strategy {strategy}")

    def _forward_brute(self, data: NNPInput):
        """
        Prepares the input tensors for passing to the model.

        This method handles general input manipulation, such as calculating
        distances and generating the pair list. It also calls the model-specific
        input preparation.

        Parameters
        ----------
        data : NNPInput
            The input data provided by the dataset, containing atomic numbers,
            positions, and other necessary information.

        Returns
        -------
        PairListOutputs
            Contains pair indices, distances (d_ij), and displacement vectors (r_ij) for atom pairs within the cutoff.
        """
        # ---------------------------
        # general input manipulation
        positions = data.positions
        atomic_subsystem_indices = data.atomic_subsystem_indices

        n = atomic_subsystem_indices.size(0)

        # avoid reinitializing indices if they are already set and haven't changed
        if self.indices.shape[0] != n:
            # Generate a range of indices from 0 to n-1
            self.indices = torch.arange(n, device=atomic_subsystem_indices.device)

            # Create a meshgrid of indices
            self.i_final_pairs, self.j_final_pairs = torch.meshgrid(
                self.indices, self.indices, indexing="ij"
            )
            # We will only consider unique pairs; for non-unique pairs we can just appropriately copy
            # the data as it will be faster than extra computations.
            mask = self.i_final_pairs < self.j_final_pairs

            self.i_final_pairs = self.i_final_pairs[mask]
            self.j_final_pairs = self.j_final_pairs[mask]

        # calculate r_ij and d_ij
        r_ij, d_ij = self.displacement_function(
            positions[self.i_final_pairs],
            positions[self.j_final_pairs],
            data.box_vectors,
            data.is_periodic,
        )
        in_cutoff = (d_ij <= self.cutoff).squeeze()
        total_pairs = in_cutoff.sum()

        if self.only_unique_pairs:
            # using this instead of torch.stack to ensure that if we only have a single pair
            # we don't run into an issue with tensor shapes.
            # note this will fail if there are no interacting pairs

            pairs = torch.zeros(
                2, total_pairs, dtype=torch.int64, device=positions.device
            )

            pairs[0] = self.i_final_pairs[in_cutoff]
            pairs[1] = self.j_final_pairs[in_cutoff]

            return PairlistData(
                pair_indices=pairs,
                d_ij=d_ij[in_cutoff],
                r_ij=r_ij[in_cutoff],
            )

        else:
            pairs_full, d_ij_full, r_ij_full = self._copy_to_nonunique(
                self.i_final_pairs[in_cutoff],
                self.j_final_pairs[in_cutoff],
                d_ij[in_cutoff],
                r_ij[in_cutoff],
                total_pairs,
            )
            return PairlistData(
                pair_indices=pairs_full,
                d_ij=d_ij_full,
                r_ij=r_ij_full,
            )

    def _forward_verlet(self, data: NNPInput):
        """
        Prepares the input tensors for passing to the model.

        This method handles general input manipulation, such as calculating
        distances and generating the pair list. It also calls the model-specific
        input preparation.

        Parameters
        ----------
        data : NNPInput
            The input data provided by the dataset, containing atomic numbers,
            positions, and other necessary information.

        Returns
        -------
        PairListOutputs
            Contains pair indices, distances (d_ij), and displacement vectors (r_ij) for atom pairs within the cutoff.
        """
        # ---------------------------
        # general input manipulation
        positions = data.positions
        atomic_subsystem_indices = data.atomic_subsystem_indices

        n = atomic_subsystem_indices.size(0)
        # if the initial build we haven't yet set box vectors so set them
        # this is necessary because we need to store them to know if we need to force a rebuild
        # because the box vectors have changed
        if self.builds == 0:
            self.box_vectors = data.box_vectors

        box_changed = torch.any(self.box_vectors != data.box_vectors)

        # avoid reinitializing indices if they are already set and haven't changed
        if self.indices.shape[0] != n:
            self.box_vectors = data.box_vectors
            self.positions_old = positions

            # self.i_pairs and self.j_pairs are all possible unique pairs in the system
            # and will need to be regenerated if the number of particles change
            self.i_pairs, self.j_pairs = self._init_verlet_pairs(n, positions.device)

            #  self.nlist_i_pairs, self.nlist_j_pairs are the pairs within the cutoff+skin
            self.nlist_i_pairs, self.nlist_j_pairs, r_ij, d_ij = (
                self._build_verlet_nlist(positions, data.box_vectors, data.is_periodic)
            )
        elif box_changed:
            # if the box vectors have changed, we need to rebuild the nlist
            # but do not need to regenerate all possible pairs (i_pairs, j_pairs)
            self.box_vectors = data.box_vectors
            self.positions_old = positions

            self.nlist_i_pairs, self.nlist_j_pairs, r_ij, d_ij = (
                self._build_verlet_nlist(positions, data.box_vectors, data.is_periodic)
            )
        elif self._check_verlet_nlist(positions, data.box_vectors, data.is_periodic):
            # if the maximum displacement exceeds half the skin distance, rebuild the nlist
            # but do not need to regenerate all possible pairs (i_pairs, j_pairs)
            self.positions_old = positions
            self.nlist_i_pairs, self.nlist_j_pairs, r_ij, d_ij = (
                self._build_verlet_nlist(positions, data.box_vectors, data.is_periodic)
            )
        else:
            # if the nlist does not need to be rebuilt, and nothing else has changed
            # we can just calculate the displacement vectors and distances
            r_ij, d_ij = self.displacement_function(
                positions[self.nlist_i_pairs],
                positions[self.nlist_j_pairs],
                data.box_vectors,
                data.is_periodic,
            )

        # identify which pairs in the neighbor list are within the cutoff
        in_cutoff = (d_ij <= self.cutoff).squeeze()
        total_pairs = in_cutoff.sum()

        # we can take advantage of the pairwise nature to just copy the unique pairs to non-unique pairs
        # copying is generally faster than the extra computations associated with considering non-unique pairs
        if self.only_unique_pairs:
            # using this approach instead of torch.stack to ensure that if we only have a single pair
            # we don't run into an issue with shapes.

            pairs = torch.zeros(
                2, total_pairs, dtype=torch.int64, device=positions.device
            )

            pairs[0] = self.nlist_i_pairs[in_cutoff]
            pairs[1] = self.nlist_j_pairs[in_cutoff]

            return PairlistData(
                pair_indices=pairs,
                d_ij=d_ij[in_cutoff],
                r_ij=r_ij[in_cutoff],
            )

        else:
            pairs_full, d_ij_full, r_ij_full = self._copy_to_nonunique(
                self.nlist_i_pairs[in_cutoff],
                self.nlist_j_pairs[in_cutoff],
                d_ij[in_cutoff],
                r_ij[in_cutoff],
                total_pairs,
            )
            return PairlistData(
                pair_indices=pairs_full,
                d_ij=d_ij_full,
                r_ij=r_ij_full,
            )


class NeighborListForTraining(torch.nn.Module):
    def __init__(self, cutoff: float, only_unique_pairs: bool = False):
        """
        Calculating the interacting pairs.  This is primarily intended for use during training,
        as this will utilize the pre-computed pair list from the dataset

        Parameters
        ----------
        cutoff : float
            The cutoff distance for neighbor list calculations.
        only_unique_pairs : bool, optional
            If True, only unique pairs are returned (default is False).
        """

        super().__init__()

        self.only_unique_pairs = only_unique_pairs
        self.pairlist = Pairlist(only_unique_pairs)
        self.register_buffer("cutoff", torch.tensor(cutoff))

    def calculate_r_ij(
        self, pair_indices: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Compute displacement vectors between atom pairs.

        Parameters
        ----------
        pair_indices : torch.Tensor
            Atom indices for pairs of atoms. Shape: [2, n_pairs].
        positions : torch.Tensor
            Atom positions. Shape: [atoms, 3].

        Returns
        -------
        torch.Tensor
            Displacement vectors between atom pairs. Shape: [n_pairs, 3].
        """
        # Select the pairs of atom coordinates from the positions
        selected_positions = positions.index_select(0, pair_indices.view(-1)).view(
            2, -1, 3
        )
        return selected_positions[1] - selected_positions[0]

    def calculate_d_ij(self, r_ij: torch.Tensor) -> torch.Tensor:
        """
        ompute Euclidean distances between atoms in each pair.

        Parameters
        ----------
        r_ij : torch.Tensor
            Displacement vectors between atoms in a pair. Shape: [n_pairs, 3].

        Returns
        -------
        torch.Tensor
            Euclidean distances. Shape: [n_pairs, 1].
        """
        return r_ij.norm(dim=1).unsqueeze(1)

    def _calculate_interacting_pairs(
        self,
        positions: torch.Tensor,
        atomic_subsystem_indices: torch.Tensor,
        pair_indices: torch.Tensor,
    ) -> PairlistData:
        """
        Compute the neighbor list considering a cutoff distance.

        Parameters
        ----------
        positions : torch.Tensor
            Atom positions. Shape: [nr_systems, nr_atoms, 3].
        atomic_subsystem_indices : torch.Tensor
            Indices identifying atoms in subsystems. Shape: [nr_atoms].
        pair_indices : torch.Tensor
            Precomputed pair indices.

        Returns
        -------
        PairListOutputs
            A dataclass containing 'pair_indices', 'd_ij' (distances), and 'r_ij' (displacement vectors).
        """

        r_ij = self.calculate_r_ij(pair_indices, positions)
        d_ij = self.calculate_d_ij(r_ij)

        in_cutoff = (d_ij <= self.cutoff).squeeze()
        # Get the atom indices within the cutoff
        pair_indices_within_cutoff = pair_indices[:, in_cutoff]

        return PairlistData(
            pair_indices=pair_indices_within_cutoff,
            d_ij=d_ij[in_cutoff],
            r_ij=r_ij[in_cutoff],
        )

    def forward(self, data: Union[NNPInput, NamedTuple]) -> PairlistData:
        """
        Compute the pair list, distances, and displacement vectors for the given
        input data.

        Parameters
        ----------
        data : Union[NNPInput, NamedTuple]
            Input data containing atomic numbers, positions, and subsystem
            indices.

        Returns
        -------
        PairlistData
            A namedtuple containing the pair indices, distances, and
            displacement vectors.
        """
        # ---------------------------
        # general input manipulation
        positions = data.positions
        atomic_subsystem_indices = data.atomic_subsystem_indices
        # calculate pairlist if it is not provided

        if data.pair_list is None or data.pair_list.shape[0] == 0:
            # note, we set the flag for unique pairs when instantiated in the constructor
            # and thus this call will return unique pairs if requested.
            pair_list = self.pairlist.enumerate_all_pairs(atomic_subsystem_indices)

        else:
            pair_list = data.pair_list

            # when we precompute the pairlist, we included all pairs, including non-unique
            # since we do this before we know about which potential we are using
            # and whether we require unique pairs or not
            # thus, if the pairlist is provided we need to remove redundant pairs if requested
            if self.only_unique_pairs:
                i_indices = pair_list[0]
                j_indices = pair_list[1]
                unique_pairs_mask = i_indices < j_indices
                i_final_pairs = i_indices[unique_pairs_mask]
                j_final_pairs = j_indices[unique_pairs_mask]
                pair_list = torch.stack((i_final_pairs, j_final_pairs))

        pairlist_output = self._calculate_interacting_pairs(
            positions=positions,
            atomic_subsystem_indices=atomic_subsystem_indices,
            pair_indices=pair_list.to(torch.int64),
        )

        return pairlist_output


#
