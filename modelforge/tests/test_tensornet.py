import pytest


def test_tensornet_init():
    from modelforge.potential.tensornet import TensorNet

    net = TensorNet()
    assert net is not None


def test_compare_radial_symmetry_features():
    # Compare the TensorNet radial symmetry function
    # to the output of the modelforge radial symmetry function

    import torch
    from openff.units import unit
    from torchmdnet.models.utils import ExpNormalSmearing

    from modelforge.potential.utils import CosineCutoff
    from modelforge.potential.utils import TensorNetRadialSymmetryFunction

    # generate a random list of distances, all < 5
    d_ij = torch.rand(5, 1) * 5 # NOTE: angstrom

    # TensorNet constants
    radial_cutoff = 5.0
    radial_start = 0.0
    radial_dist_divisions = 8

    rsf = TensorNetRadialSymmetryFunction(
        number_of_radial_basis_functions=radial_dist_divisions,
        max_distance=radial_cutoff * unit.angstrom,
        min_distance=radial_start * unit.angstrom,
    )
    r_mf = rsf(d_ij / 10)  # torch.Size([5, 1, 8]) # NOTE: nanometer
    cutoff_module = CosineCutoff(radial_cutoff * unit.angstrom)

    rcut_ij = cutoff_module(d_ij.unsqueeze(-1) / 10)  # torch.Size([5]) # NOTE: nanometer
    r_mf = r_mf * rcut_ij

    rsf_tn = ExpNormalSmearing(
        cutoff_lower=radial_start,
        cutoff_upper=radial_cutoff,
        num_rbf=radial_dist_divisions,
        trainable=False,
    )
    r_tn = rsf_tn(d_ij)

    assert torch.allclose(r_mf, r_tn)


def test_representation():
    import torch
    from openff.units import unit
    from torch import nn
    from torchmdnet.models.model import create_model, load_model
    from torchmdnet.models.tensornet import TensorEmbedding
    from torchmdnet.models.utils import ExpNormalSmearing, OptimizedDistance
    from modelforge.potential.tensornet import TensorNetRepresentation

    num_atoms = 5
    hidden_channels = 2
    num_rbf = 8
    act_class = nn.SiLU
    cutoff_lower = 0.0
    cutoff_upper = 5.0
    trainable_rbf = False
    max_z = 128
    dtype = torch.float32

    ################ TensorNet ################
    def create_example_batch(n_atoms=6, multiple_batches=True):
        zs = torch.tensor([1, 6, 7, 8, 9], dtype=torch.long)
        z = zs[torch.randint(0, len(zs), (n_atoms,))]

        pos = torch.randn(len(z), 3)

        batch = torch.zeros(len(z), dtype=torch.long)
        if multiple_batches:
            batch[len(batch) // 2 :] = 1
        return z, pos, batch

    # TensorNet embedding modules setup
    tensor_embedding = TensorEmbedding(
        hidden_channels,
        num_rbf,
        act_class,
        cutoff_lower,
        cutoff_upper,
        trainable_rbf,
        max_z,
        dtype,
    )
    tensor_embedding.reset_parameters()
    distance_module = OptimizedDistance(
        cutoff_lower,
        cutoff_upper,
        max_num_pairs=-64,
        return_vecs=True,
        loop=True,
        check_errors=True,
        resize_to_fit=False,
        box=None,
        long_edge_index=True,
    )
    distance_expansion = ExpNormalSmearing(
        cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
    )

    # calculate embedding
    z, pos, batch = create_example_batch()
    edge_index, edge_weight, edge_vec = distance_module(pos, batch, None)

    # static shape
    q = torch.zeros_like(z, device=z.device, dtype=z.dtype)
    zp = z
    mask = (edge_index[0] < 0).unsqueeze(0).expand_as(edge_index)
    zp = torch.cat((z, torch.zeros(1, device=z.device, dtype=z.dtype)), dim=0)
    q = torch.cat((q, torch.zeros(1, device=q.device, dtype=q.dtype)), dim=0)
    # I trick the model into thinking that the masked edges pertain to the extra atom
    # WARNING: This can hurt performance if max_num_pairs >> actual_num_pairs
    edge_index = edge_index.masked_fill(mask, z.shape[0])
    edge_weight = edge_weight.masked_fill(mask[0], 0)
    edge_vec = edge_vec.masked_fill(
        mask[0].unsqueeze(-1).expand_as(edge_vec), 0
    )

    edge_attr = distance_expansion(edge_weight)
    X_tn = tensor_embedding(zp, edge_index, edge_weight, edge_vec, edge_attr)
    ################ TensorNet ################

if __name__ == "__main__":
    import torch
    torch.manual_seed(0)
    test_compare_radial_symmetry_features()

    test_representation()
