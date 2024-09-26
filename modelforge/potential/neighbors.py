import torch
from loguru import logger as log

from modelforge.dataset.dataset import NNPInputTuple


class OrthogonalDisplacementFunction(torch.nn.Module):
    def __init__(self, periodic: bool):
        """
        Compute displacement vectors between pairs of atoms, considering periodic boundary conditions.

        Attributes
        ----------
        periodic : bool
            Whether to apply periodic boundary conditions.
        """
        super().__init__()

        self.register_buffer("periodic", torch.tensor(periodic))

    def forward(
        self,
        coordinate_i: torch.Tensor,
        coordinate_j: torch.Tensor,
        box_vectors: torch.Tensor,
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

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Displacement vectors (r_ij) of shape [n_pairs, 3] and distances (d_ij) of shape [n_pairs, 1].
        """
        r_ij = coordinate_i - coordinate_j

        if self.periodic == True:
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

    def forward(self, data: NNPInputTuple):
        """
        Prepares the input tensors for passing to the model.

        This method handles general input manipulation, such as calculating
        distances and generating the pair list. It also calls the model-specific
        input preparation.

        Parameters
        ----------
        data : NNPInputTuple
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
        )

        in_cutoff = (d_ij <= self.cutoff).squeeze()

        if self.only_unique_pairs:
            return PairlistData(
                pair_indices=torch.stack(
                    [self.i_final_pairs[in_cutoff], self.j_final_pairs[in_cutoff]]
                ),
                d_ij=d_ij[in_cutoff],
                r_ij=r_ij[in_cutoff],
            )

        else:

            total_pairs = in_cutoff.sum()

            r_ij_full = torch.zeros(
                total_pairs * 2, 3, dtype=positions.dtype, device=positions.device
            )
            d_ij_full = torch.zeros(
                total_pairs * 2, 1, dtype=positions.dtype, device=positions.device
            )

            temp = r_ij[in_cutoff]
            # i, j pairs
            r_ij_full[0:total_pairs] = temp
            # j, i pairs, require we swap the sign
            r_ij_full[total_pairs : 2 * total_pairs] = -temp

            del r_ij, temp

            temp = d_ij[in_cutoff]
            d_ij_full[0:total_pairs] = temp
            d_ij_full[total_pairs : 2 * total_pairs] = temp

            del d_ij, temp

            temp1 = self.i_pairs[in_cutoff]
            temp2 = self.j_pairs[in_cutoff]

            pairs = torch.zeros(
                2, total_pairs * 2, dtype=torch.int64, device=positions.device
            )

            pairs[0][0:total_pairs] = temp1
            pairs[1][0:total_pairs] = temp2
            pairs[0][total_pairs : 2 * total_pairs] = temp2
            pairs[1][total_pairs : 2 * total_pairs] = temp1

            del temp1, temp2

            return PairlistData(
                pair_indices=pairs,
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
        self.register_buffer("skin", torch.tensor(skin))
        self.register_buffer("cutoff_plus_skin", torch.tensor(cutoff + skin))
        self.register_buffer("only_unique_pairs", torch.tensor(only_unique_pairs))

        self.displacement_function = displacement_function
        self.indices = torch.tensor([])
        self.i_final_pairs = torch.tensor([])
        self.j_final_pairs = torch.tensor([])

        self.positions_old = torch.tensor([])
        self.nlist_pairs = torch.tensor([])
        self.builds = 0

        log.info("Initializing Verlet Neighborlist with N^2 building routine.")

    def _check_nlist(self, positions: torch.Tensor, box_vectors: torch.Tensor):
        r_ij, d_ij = self.displacement_function(
            self.positions_old, positions, box_vectors
        )

        if torch.any(d_ij > self.skin * 0.5):
            return True
        else:
            return False

    def _init_pairs(self, n_particles: int, device: torch.device):
        self.particle_ids = torch.arange(n_particles, device=device)

        self.i_pairs, self.j_pairs = torch.meshgrid(
            self.particle_ids,
            self.particle_ids,
            indexing="ij",
        )

        mask = self.i_pairs < self.j_pairs
        self.i_pairs = self.i_pairs[mask]
        self.j_pairs = self.j_pairs[mask]

    def _build_nlist(self, positions: torch.Tensor, box_vectors: torch.Tensor):

        r_ij, d_ij = self.displacement_function(
            positions[self.i_pairs], positions[self.j_pairs], box_vectors
        )

        in_cutoff = (d_ij < self.cutoff_plus_skin).squeeze()
        self.nlist_pairs = torch.stack(
            [self.i_pairs[in_cutoff], self.j_pairs[in_cutoff]]
        )
        self.builds += 1
        return r_ij[in_cutoff], d_ij[in_cutoff]

    def forward(self, data: NNPInputTuple):
        """
        Prepares the input tensors for passing to the model.

        This method handles general input manipulation, such as calculating
        distances and generating the pair list. It also calls the model-specific
        input preparation.

        Parameters
        ----------
        data : NNPInputTuple
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
            self.positions_old = positions
            self.init_pairs(n, positions.device)
            r_ij, d_ij = self._build_nlist(positions, data.box_vectors)
        elif self._check_nlist(positions, data.box_vectors):
            self.positions_old = positions
            r_ij, d_ij = self._build_nlist(positions, data.box_vectors)
        else:
            r_ij, d_ij = self.displacement_function(
                positions[self.nlist_pairs[0]],
                positions[self.nlist_pairs[1]],
                data.box_vectors,
            )

        in_cutoff = (d_ij <= self.cutoff).squeeze()

        if self.only_unique_pairs:
            return PairlistData(
                pair_indices=torch.stack(
                    [self.nlist_pairs[0][in_cutoff], self.nlist_pairs[1][in_cutoff]]
                ),
                d_ij=d_ij[in_cutoff],
                r_ij=r_ij[in_cutoff],
            )

        else:

            total_pairs = in_cutoff.sum()

            r_ij_full = torch.zeros(
                total_pairs * 2, 3, dtype=positions.dtype, device=positions.device
            )
            d_ij_full = torch.zeros(
                total_pairs * 2, 1, dtype=positions.dtype, device=positions.device
            )

            temp = r_ij[in_cutoff]
            r_ij_full[0:total_pairs] = temp
            r_ij_full[total_pairs : 2 * total_pairs] = -temp

            del r_ij, temp

            temp = d_ij[in_cutoff]
            d_ij_full[0:total_pairs] = temp
            d_ij_full[total_pairs : 2 * total_pairs] = temp

            del d_ij, temp

            temp1 = self.nlist_pairs[0][in_cutoff]
            temp2 = self.nlist_pairs[1][in_cutoff]

            pairs = torch.zeros(
                2, total_pairs * 2, dtype=torch.int64, device=positions.device
            )

            pairs[0][0:total_pairs] = temp1
            pairs[1][0:total_pairs] = temp2
            pairs[0][total_pairs : 2 * total_pairs] = temp2
            pairs[1][total_pairs : 2 * total_pairs] = temp1

            del temp1, temp2

            return PairlistData(
                pair_indices=pairs,
                d_ij=d_ij_full,
                r_ij=r_ij_full,
            )


# As this currently isn't any faster or have an appreciable memory footprint than the PyTorch implementation,
# this is currently commented out.  This will be revisited in the future as the NNPOps library is updated.
#
# from NNPOps.neighbors import getNeighborPairs
#
#
# class NeighborlistVerletNsqNNPOps(torch.nn.Module):
#     """
#     Verlet neighbor list calculation for inference implemented using NNPOps backend.
#
#     Rebuilding of the neighborlist still uses an N^2 approach for initial construction, but uses the NNPOps CUDA kernel.
#     Rebuilding occurs when the maximum displacement of any particle exceeds half the skin distance.

#
#     """
#
#     def __init__(
#         self,
#         cutoff: float,
#         skin: float,
#         displacement_function: OrthogonalDisplacementFunction,
#         only_unique_pairs: bool = False,
#     ):
#         """
#         Compute neighbor lists for inference, filtering pairs based on a cutoff distance.
#
#         Parameters
#         ----------
#         cutoff : float
#             The cutoff distance for neighbor list calculations.
#         skin : float
#             The skin distance for neighbor list calculations.
#         displacement_function : OrthogonalDisplacementFunction
#             The function to calculate displacement vectors and distances between atom pairs, taking into account
#             the specified boundary conditions.
#         only_unique_pairs : bool, optional
#             Whether to only use unique pairs in the pair list calculation, by
#             default True. This should be set to True for all message passing
#             networks.
#         """
#
#         super().__init__()
#
#         self.register_buffer("cutoff", torch.tensor(cutoff))
#         self.register_buffer("skin", torch.tensor(skin))
#         self.register_buffer("cutoff_plus_skin", torch.tensor(cutoff + skin))
#         self.register_buffer("only_unique_pairs", torch.tensor(only_unique_pairs))
#
#         self.displacement_function = displacement_function
#         self.indices = torch.tensor([])
#         self.i_final_pairs = torch.tensor([])
#         self.j_final_pairs = torch.tensor([])
#
#         self.positions_old = torch.tensor([])
#         self.nlist_pairs = torch.tensor([])
#         self.builds = 0
#
#         log.debug("Initializing Verlet Neighborlist with N^2 building routine.")
#
#     def _check_nlist(self, positions: torch.Tensor, box_vectors: torch.Tensor):
#         r_ij, d_ij = self.displacement_function(
#             self.positions_old, positions, box_vectors
#         )
#
#         if torch.any(d_ij > self.skin * 0.5):
#             return True
#         else:
#             return False
#
#     def _build_nlist(self, positions: torch.Tensor, box_vectors: torch.Tensor):
#
#         if self.periodic == True:
#             neighbors, r_ij_temp, d_ij_temp, _ = getNeighborPairs(
#                 positions,
#                 cutoff=self.cutoff,
#                 max_num_pairs=-1,
#                 check_errors=False,
#                 box_vectors=box_vectors,
#             )
#         else:
#             neighbors, r_ij_temp, d_ij_temp, _ = getNeighborPairs(
#                 positions,
#                 cutoff=self.cutoff,
#                 max_num_pairs=-1,
#                 check_errors=False,
#             )
#
#         _, indices = torch.sort(neighbors[1], descending=True)
#         del _
#
#         sorted_neighbors = neighbors[:, indices]
#
#         d_ij = d_ij_temp[indices].unsqueeze(1)
#         r_ij = r_ij_temp[indices]
#
#         in_cutoff = (d_ij < self.cutoff_skin).squeeze()
#
#         self.nlist_pairs = torch.stack(
#             [sorted_neighbors[0][in_cutoff], sorted_neighbors[1][in_cutoff]]
#         )
#         self.builds += 1
#         return r_ij[in_cutoff], d_ij[in_cutoff]
#
#     def forward(self, data: NNPInputTuple):
#         """
#         Prepares the input tensors for passing to the model.
#
#         This method handles general input manipulation, such as calculating
#         distances and generating the pair list. It also calls the model-specific
#         input preparation.
#
#         Parameters
#         ----------
#         data : NNPInputTuple
#             The input data provided by the dataset, containing atomic numbers,
#             positions, and other necessary information.
#
#         Returns
#         -------
#         PairListOutputs
#             Contains pair indices, distances (d_ij), and displacement vectors (r_ij) for atom pairs within the cutoff.
#         """
#         # ---------------------------
#         # general input manipulation
#         positions = data.positions
#         atomic_subsystem_indices = data.atomic_subsystem_indices
#
#         n = atomic_subsystem_indices.size(0)
#
#         # avoid reinitializing indices if they are already set and haven't changed
#         if self.indices.shape[0] != n or self.builds == 0:
#             self.positions_old = positions
#             r_ij, d_ij = self._build_nlist(positions, data.box_vectors)
#         elif self._check_nlist(positions, data.box_vectors):
#             self.positions_old = positions
#             r_ij, d_ij = self._build_nlist(positions, data.box_vectors)
#         else:
#             r_ij, d_ij = self.displacement_function(
#                 positions[self.nlist_pairs[0]],
#                 positions[self.nlist_pairs[1]],
#                 data.box_vectors,
#             )
#
#         in_cutoff = (d_ij <= self.cutoff).squeeze()
#
#         if self.only_unique_pairs:
#             return PairlistData(
#                 pair_indices=torch.stack(
#                     [self.nlist_pairs[0][in_cutoff], self.nlist_pairs[1][in_cutoff]]
#                 ),
#                 d_ij=d_ij[in_cutoff],
#                 r_ij=r_ij[in_cutoff],
#             )
#
#         else:
#
#             total_pairs = in_cutoff.sum()
#
#             r_ij_full = torch.zeros(
#                 total_pairs * 2, 3, dtype=positions.dtype, device=positions.device
#             )
#             d_ij_full = torch.zeros(
#                 total_pairs * 2, 1, dtype=positions.dtype, device=positions.device
#             )
#
#             temp = r_ij[in_cutoff]
#             r_ij_full[0:total_pairs] = temp
#             r_ij_full[total_pairs : 2 * total_pairs] = -temp
#
#             del r_ij, temp
#
#             temp = d_ij[in_cutoff]
#             d_ij_full[0:total_pairs] = temp
#             d_ij_full[total_pairs : 2 * total_pairs] = temp
#
#             del d_ij, temp
#
#             temp1 = self.nlist_pairs[0][in_cutoff]
#             temp2 = self.nlist_pairs[1][in_cutoff]
#
#             pairs = torch.zeros(
#                 2, total_pairs * 2, dtype=torch.int64, device=positions.device
#             )
#
#             pairs[0][0:total_pairs] = temp1
#             pairs[1][0:total_pairs] = temp2
#             pairs[0][total_pairs : 2 * total_pairs] = temp2
#             pairs[1][total_pairs : 2 * total_pairs] = temp1
#
#             del temp1, temp2
#
#             return PairlistData(
#                 pair_indices=pairs,
#                 d_ij=d_ij_full,
#                 r_ij=r_ij_full,
#             )
