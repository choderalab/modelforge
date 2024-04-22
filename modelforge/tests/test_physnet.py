def test_physnet_init():

    from modelforge.potential.physnet import PhysNet

    model = PhysNet()


def test_physnet_forward():

    from modelforge.potential.physnet import PhysNet
    from modelforge.dataset.qm9 import QM9Dataset
    from modelforge.dataset.dataset import TorchDataModule
    from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy

    # Set up dataset
    data = QM9Dataset(for_unit_testing=True)
    dataset = TorchDataModule(
        data, batch_size=1, splitting_strategy=FirstComeFirstServeSplittingStrategy()
    )

    dataset.prepare_data(remove_self_energies=True, normalize=False)
    # get methane input
    methane = next(iter(dataset.train_dataloader())).nnp_input

    model = PhysNet()
    model(methane)


def test_rbf_equivalence():
    import tensorflow as tf
    import numpy as np
    import torch

    number_of_radial_basis_functions = K = 20
    cutoff = _unitless_max_distance = 5.0
    _unitless_min_distance = 0.0

    # PhysNet implementation
    # width
    def softplus_inverse(x):
        return x + np.log(-np.expm1(-x))

    widths = [softplus_inverse((0.5 / ((1.0 - np.exp(-cutoff)) / K)) ** 2)] * K
    _widths = tf.nn.softplus(tf.Variable(np.asarray(widths), name="widths"))
    pn_widths = _widths.numpy()

    # Modelforge implementation
    # width
    start_value = torch.exp(
        torch.scalar_tensor(-_unitless_max_distance + _unitless_min_distance)
    )
    radial_scale_factor = torch.tensor(
        [(2 / number_of_radial_basis_functions * (1 - start_value)) ** -2]
        * number_of_radial_basis_functions
    )
    mf_widths = radial_scale_factor.numpy()

    assert np.allclose(pn_widths, mf_widths)

    # PhysNet implementation
    # center_position
    centers = softplus_inverse(np.linspace(1.0, np.exp(-cutoff), K))
    _centers = tf.nn.softplus(tf.Variable(np.asarray(centers), name="centers"))
    pn_centers = _centers.numpy()

    start_value = torch.exp(
        torch.scalar_tensor(-_unitless_max_distance + _unitless_min_distance)
    )
    centers = torch.linspace(start_value, 1, number_of_radial_basis_functions)

    mf_centers = centers.numpy()
    assert np.allclose(
        np.flip(pn_centers), mf_centers
    )  # NOTE: The PhysNet implementation uses the reverse order of the centers
