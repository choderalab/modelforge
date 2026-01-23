from modelforge.curate import create_dataset_from_hdf5
from modelforge.utils.units import chem_context
from openff.units import unit
from tqdm import tqdm

ase = {
    "H": -1576.5513678678228 * unit.kilojoule_per_mole,
    "Li": -19221.76009670645 * unit.kilojoule_per_mole,
    "C": -100114.38959681295 * unit.kilojoule_per_mole,
    "N": -143829.94579288512 * unit.kilojoule_per_mole,
    "O": -197627.70305727186 * unit.kilojoule_per_mole,
    "F": -262291.3177197502 * unit.kilojoule_per_mole,
    "Na": -425714.1444283384 * unit.kilojoule_per_mole,
    "Mg": -523447.29044746497 * unit.kilojoule_per_mole,
    "P": -896460.9044578229 * unit.kilojoule_per_mole,
    "S": -1045607.5830439369 * unit.kilojoule_per_mole,
    "Cl": -1208414.168327362 * unit.kilojoule_per_mole,
    "K": -1574847.955709633 * unit.kilojoule_per_mole,
    "Ca": -1777543.2887296947 * unit.kilojoule_per_mole,
    "Br": -6758454.442850963 * unit.kilojoule_per_mole,
    "I": -781842.6578771132 * unit.kilojoule_per_mole,
}

ase2 = {
    "Br": -2574.1167240829964 * unit.hartree,
    "C": -37.87264507233593 * unit.hartree,
    "Ca": -676.9528465198214 * unit.hartree,
    "Cl": -460.1988762285739 * unit.hartree,
    "F": -99.78611622985483 * unit.hartree,
    "H": -0.498760510048753 * unit.hartree,
    "I": -297.76228914445625 * unit.hartree,
    "K": -599.8025677513111 * unit.hartree,
    "Li": -7.285254714046546 * unit.hartree,
    "Mg": -199.2688420040449 * unit.hartree,
    "N": -54.62327513368922 * unit.hartree,
    "Na": -162.11366478783253 * unit.hartree,
    "O": -75.11317840410095 * unit.hartree,
    "P": -341.3059197024934 * unit.hartree,
    "S": -398.1599636677874 * unit.hartree,
}

ase_ph = {
    "H": -1596.6973305434612 * unit.kilojoule_per_mole,
    "C": -100059.79872980758 * unit.kilojoule_per_mole,
    "O": -197491.36594960644 * unit.kilojoule_per_mole,
}


def calculate_reference_energy(atomic_numbers, ase):
    from modelforge.dataset.utils import _ATOMIC_NUMBER_TO_ELEMENT

    atomic_numbers = list(atomic_numbers.reshape(-1))
    # sum up the reference energy for each element in the atomic numbers
    reference_energy = [
        ase[_ATOMIC_NUMBER_TO_ELEMENT[atomic_number]]
        for atomic_number in atomic_numbers
    ]

    return sum(reference_energy)


# # load up the dataset
# dataset = create_dataset_from_hdf5(
#     "~/modelforge_testing_dataset_cache/spice_1_dataset_v1.1.hdf5", "spice_1", read_from_local_db=True
# )
# load up the dataset
dataset = create_dataset_from_hdf5(
    "~/modelforge_testing_dataset_cache/PhAlkEthOH_openff_dataset_v1.1.hdf5",
    "ph1",
    read_from_local_db=False,
)
energies = []
# loop over the records and calculate reference energies
for record_name in tqdm(dataset.keys()):
    record = dataset.get_record(record_name)
    atomic_numbers = record.get_property("atomic_numbers").value
    reference_energy = calculate_reference_energy(atomic_numbers, ase_ph).to(
        unit.kilojoule_per_mole, "chem"
    )
    energy_prop = record.get_property("dft_total_energy")
    # hdf5 file in general will be storing energies in kJ/mol already, but in case they are in other units
    # such as hartree, we will just convert them to kJ/mol here, so that we can specify the "chem" chemical context
    # used to make this conversion
    energy = (energy_prop.value * energy_prop.units).to(unit.kilojoule_per_mole, "chem")

    corrected_energy = energy - reference_energy

    # add the magnitude of the energies to the list
    energies.append(corrected_energy.to(unit.kilojoule_per_mole).magnitude)

from matplotlib import pyplot as plt
import numpy as np

plt.hist(np.concatenate(energies), bins=100)
plt.xlabel("DFT Total Energy - Reference Energy (kJ/mol)")
plt.ylabel("Count")
plt.title("Histogram of Corrected DFT Total Energies in phalkethoh Dataset")
plt.savefig("phalkethoh_corrected_energy_dist2.png")
