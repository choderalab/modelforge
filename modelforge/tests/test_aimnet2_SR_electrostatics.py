"""
Validate the Gaussian-smeared electrostatics of the AimNet2SR core.

The core computes the potential and energy of a set of atomic charges with the
kernel erf(r / (sqrt(2) sigma)) / r plus an analytic self term sqrt(2/pi) / sigma
(``spin_resolved_gaussian_smeared_potential``). This is the exact electrostatic
energy of spherical Gaussian charges of width sigma / sqrt(2), including their
self energies. The tests check this against independent references (numerical
quadrature over the Gaussian densities, a brute-force pair loop, bare Coulomb in
the far field and ``CoulombPotential``) and check the core's reported energy
against its own charges.
"""

import math

import numpy as np
import pytest
import torch

from modelforge.potential.aimnet2_SR import spin_resolved_gaussian_smeared_potential
from modelforge.potential.processing import CoulombPotential

SIGMA = 0.1  # nm, AimNet2SRCore default electrostatic_smearing_width
K_E_CODE = 138.96  # kJ/mol nm e^-2, as hard-coded in CoulombPotential
ES_CUTOFF = 1.5  # nm, aimnet2_sr default electrostatic_maximum_interaction_radius
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _full_pair_list(positions: torch.Tensor, system_indices: torch.Tensor):
    """All ordered pairs (i, j), i != j, within the same system."""
    n = positions.shape[0]
    i, j = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    mask = (i != j) & (system_indices[i] == system_indices[j])
    idx = torch.stack([i[mask], j[mask]])
    d_ij = (positions[idx[1]] - positions[idx[0]]).norm(dim=-1, keepdim=True)
    return idx, d_ij


def _gaussian_potential_quadrature(r: np.ndarray, s: float) -> np.ndarray:
    """
    Potential at distance r from a unit spherical Gaussian charge of width s,
    by Gauss's law with radial quadrature (no erf used):
    phi(r) = Q_enc(r) / r + int_r^inf 4 pi r' rho(r') dr'.
    """
    grid = np.linspace(0.0, 14.0 * s, 200001)
    rho = np.exp(-(grid**2) / (2 * s**2)) / (2 * np.pi * s**2) ** 1.5
    shell = 4 * np.pi * grid**2 * rho
    outer = 4 * np.pi * grid * rho
    q_enc = np.concatenate(
        [[0.0], np.cumsum(0.5 * (shell[1:] + shell[:-1]) * np.diff(grid))]
    )
    outer_cum = np.concatenate(
        [[0.0], np.cumsum(0.5 * (outer[1:] + outer[:-1]) * np.diff(grid))]
    )
    q_enc_r = np.interp(r, grid, q_enc)
    outer_r = outer_cum[-1] - np.interp(r, grid, outer_cum)
    return q_enc_r / r + outer_r


def _kernel(r):
    return torch.erf(r / (math.sqrt(2.0) * SIGMA)) / r


def test_kernel_is_potential_of_gaussian_of_width_sigma():
    """erf(r/sqrt(2) sigma)/r equals the Gauss's-law potential of a width-sigma
    Gaussian, i.e. the interaction of two Gaussians of width sigma/sqrt(2)."""
    r = np.array([0.02, 0.05, 0.1, 0.15, 0.3, 0.6, 1.2])
    reference = _gaussian_potential_quadrature(r, SIGMA)
    kernel = _kernel(torch.tensor(r, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(kernel, reference, rtol=1e-6)


def test_pair_energy_of_two_gaussians_matches_kernel():
    """Direct 2D quadrature of int rho_a(x) phi_b(|x - R|) d^3x for two
    Gaussians of width sigma/sqrt(2) reproduces the kernel at separation R."""
    s = SIGMA / math.sqrt(2.0)
    r = np.linspace(0.0, 10.0 * s, 1601)[1:]
    theta = np.linspace(0.0, np.pi, 801)
    rr, tt = np.meshgrid(r, theta, indexing="ij")
    rho_a = np.exp(-(rr**2) / (2 * s**2)) / (2 * np.pi * s**2) ** 1.5
    weight = 2 * np.pi * rr**2 * np.sin(tt) * rho_a
    for R in [0.03, 0.1, 0.2, 0.5]:
        dist = np.sqrt(rr**2 + R**2 - 2 * rr * R * np.cos(tt))
        dist = np.maximum(dist, 1e-12)
        phi_b = _gaussian_potential_quadrature(dist.ravel(), s).reshape(dist.shape)
        energy = _trapz(_trapz(weight * phi_b, theta, axis=1), r)
        expected = _kernel(torch.tensor(R, dtype=torch.float64)).item()
        assert energy == pytest.approx(expected, rel=2e-4)


def test_self_term_is_gaussian_self_energy_and_kernel_limit():
    """The self coefficient sqrt(2/pi)/sigma is the r -> 0 limit of the kernel
    and twice the self energy of a unit Gaussian of width sigma/sqrt(2)."""
    s = SIGMA / math.sqrt(2.0)
    coefficient = math.sqrt(2.0 / math.pi) / SIGMA

    small_r = torch.tensor([1e-6 * SIGMA], dtype=torch.float64)
    assert _kernel(small_r).item() == pytest.approx(coefficient, rel=1e-9)

    grid = np.linspace(1e-9, 14.0 * s, 200001)
    rho = np.exp(-(grid**2) / (2 * s**2)) / (2 * np.pi * s**2) ** 1.5
    phi = _gaussian_potential_quadrature(grid, s)
    self_energy = 0.5 * _trapz(4 * np.pi * grid**2 * rho * phi, grid)
    assert self_energy == pytest.approx(0.5 * coefficient, rel=1e-5)


def test_potential_matches_brute_force_pair_loop():
    """The scatter-based potential equals an explicit loop with math.erf, for a
    batch of two systems and the full (i != j) pair list."""
    torch.manual_seed(0)
    positions = torch.rand(9, 3, dtype=torch.float64) * 0.6
    system_indices = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1])
    charges = torch.randn(9, 1, dtype=torch.float64) * 0.4
    idx, d_ij = _full_pair_list(positions, system_indices)

    v = spin_resolved_gaussian_smeared_potential(d_ij, idx, 9, charges, SIGMA)

    expected = torch.zeros(9, 1, dtype=torch.float64)
    for i in range(9):
        total = math.sqrt(2.0 / math.pi) / SIGMA * charges[i].item()
        for j in range(9):
            if i == j or system_indices[i] != system_indices[j]:
                continue
            r = (positions[i] - positions[j]).norm().item()
            total += charges[j].item() * math.erf(r / (math.sqrt(2) * SIGMA)) / r
        expected[i] = total
    torch.testing.assert_close(v, expected, rtol=1e-12, atol=1e-12)


def test_unique_pair_list_gives_a_different_potential():
    """The potential (and the 0.5 in the energy) assume the full pair list; a
    unique (i < j) list drops half of the pair contributions."""
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=torch.float64)
    charges = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    idx, d_ij = _full_pair_list(positions, torch.tensor([0, 0]))
    full = spin_resolved_gaussian_smeared_potential(d_ij, idx, 2, charges, SIGMA)
    unique = idx[0] < idx[1]
    half = spin_resolved_gaussian_smeared_potential(
        d_ij[unique], idx[:, unique], 2, charges, SIGMA
    )
    assert not torch.allclose(full, half)


@pytest.mark.parametrize("r", [0.8, 1.1, 1.4])
def test_far_field_energy_matches_bare_coulomb_and_coulomb_potential(r):
    """For r >> sigma the Gaussian pair energy (energy minus self terms) equals
    bare Coulomb, and equals CoulombPotential once r > cutoff / 2, where the
    PhysNet damping is off."""
    positions = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=torch.float64)
    charges = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    system_indices = torch.tensor([0, 0])
    idx, d_ij = _full_pair_list(positions, system_indices)

    v = spin_resolved_gaussian_smeared_potential(d_ij, idx, 2, charges, SIGMA)
    energy = (0.5 * charges * v).sum()
    self_energy = (0.5 * math.sqrt(2.0 / math.pi) / SIGMA * charges**2).sum()
    gaussian_pair = K_E_CODE * (energy - self_energy).item()

    bare = K_E_CODE * charges[0].item() * charges[1].item() / r
    assert gaussian_pair == pytest.approx(bare, rel=1e-8)

    coulomb = CoulombPotential(cutoff=ES_CUTOFF)
    out = coulomb(
        {
            "electrostatic_pair_indices": idx,
            "electrostatic_d_ij": d_ij,
            "atomic_subsystem_indices": system_indices,
            "per_atom_charge": charges,
        }
    )
    assert out["per_system_electrostatic_energy"].item() == pytest.approx(
        gaussian_pair, rel=1e-8
    )


def _build_sr_potential(postprocessing: bool):
    from modelforge.potential import NeuralNetworkPotentialFactory
    from modelforge.potential.parameters import (
        ElectrostaticPotential,
        SumPerSystemEnergy,
    )
    from modelforge.utils.misc import load_configs_into_pydantic_models
    from openff.units import unit

    config = load_configs_into_pydantic_models("aimnet2_sr", "qm9")
    if postprocessing:
        pp = config["potential"].postprocessing_parameter
        pp.properties_to_process += [
            "per_system_electrostatic_energy",
            "sum_per_system_energy",
        ]
        pp.per_system_electrostatic_energy = ElectrostaticPotential(
            electrostatic_strategy="coulomb",
            maximum_interaction_radius=15.0 * unit.angstrom,
        )
        pp.sum_per_system_energy = SumPerSystemEnergy(
            contributions=["per_system_electrostatic_energy"]
        )
    return NeuralNetworkPotentialFactory.generate_potential(
        potential_parameter=config["potential"],
        training_parameter=config["training"],
        dataset_parameter=config["dataset"],
        potential_seed=42,
        use_training_mode_neighborlist=True,
    )


def _water_and_methanol():
    from modelforge.utils.prop import NNPInput

    positions = torch.tensor(
        [
            [0.0000, 0.0000, 0.0000],
            [0.0957, 0.0000, 0.0000],
            [-0.0240, 0.0927, 0.0000],
            [0.5000, 0.0000, 0.0000],
            [0.6430, 0.0000, 0.0000],
            [0.4640, 0.1030, 0.0000],
            [0.4640, -0.0510, 0.0890],
            [0.4640, -0.0510, -0.0890],
            [0.6750, -0.0910, 0.0000],
        ],
        dtype=torch.float32,
    )
    return NNPInput(
        atomic_numbers=torch.tensor([8, 1, 1, 6, 8, 1, 1, 1, 1]),
        positions=positions,
        atomic_subsystem_indices=torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1]),
        per_system_total_charge=torch.tensor([[0], [0]], dtype=torch.int32),
    )


def _gaussian_energy_reference(positions, charges, system_indices, cutoff):
    """Brute-force 0.5 sum_i q_i v_i in float64 from the core's charges."""
    positions = positions.double()
    charges = charges.double()
    idx, d_ij = _full_pair_list(positions, system_indices)
    keep = d_ij.squeeze(-1) < cutoff
    v = spin_resolved_gaussian_smeared_potential(
        d_ij[keep], idx[:, keep], positions.shape[0], charges, SIGMA
    )
    return 0.5 * charges * v


def test_core_electrostatic_energy_is_consistent_with_its_charges():
    """The core's per_atom_electrostatic_energy equals 0.5 q_i v_i recomputed
    from its own output charges."""
    potential = _build_sr_potential(postprocessing=False)
    data = _water_and_methanol()
    pairlist = potential.neighborlist.forward(data)
    out = potential.core_network.forward(
        data, pairlist.local_cutoff, pairlist.electrostatic_cutoff
    )
    expected = _gaussian_energy_reference(
        data.positions,
        out["per_atom_charge"].detach(),
        data.atomic_subsystem_indices,
        ES_CUTOFF,
    )
    torch.testing.assert_close(
        out["per_atom_electrostatic_energy"].detach().double(),
        expected,
        rtol=1e-4,
        atol=1e-6,
    )


@pytest.mark.xfail(
    strict=True,
    reason="PerAtomEnergy overwrites the CoulombPotential result with the "
    "core's unscaled Gaussian energy (processing.py, per_atom_electrostatic_energy).",
)
def test_coulomb_postprocessing_energy_reaches_total_energy():
    """With coulomb + sum_per_system_energy enabled, the electrostatic energy
    added to the total should be the CoulombPotential value."""
    potential = _build_sr_potential(postprocessing=True)
    data = _water_and_methanol()
    output = potential(data)

    pairlist = potential.neighborlist.forward(data)
    coulomb = CoulombPotential(cutoff=ES_CUTOFF)
    expected = coulomb(
        {
            "electrostatic_pair_indices": pairlist.electrostatic_cutoff.pair_indices,
            "electrostatic_d_ij": pairlist.electrostatic_cutoff.d_ij,
            "atomic_subsystem_indices": data.atomic_subsystem_indices,
            "per_atom_charge": output["per_atom_charge"].detach(),
        }
    )["per_system_electrostatic_energy"]
    torch.testing.assert_close(
        output["per_system_electrostatic_energy"].detach(),
        expected,
        rtol=1e-4,
        atol=1e-6,
    )
