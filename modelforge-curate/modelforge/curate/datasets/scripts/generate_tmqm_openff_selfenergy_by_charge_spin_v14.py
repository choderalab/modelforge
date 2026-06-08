from modelforge.curate import create_dataset_from_hdf5, Record, SourceDataset
from modelforge.curate.properties import *

import numpy as np
from openff.units import unit
from loguru import logger

# This script will read the tmqm openff dataset and only keep neutral molecules

logger.info("Loading TMQM OpenFF dataset...")

property_map = {
    "atomic_numbers": AtomicNumbers,
    "positions": Positions,
    "total_charge": TotalCharge,
    "per_system_spin_multiplicity": SpinMultiplicitiesPerSystem,
    "dft_total_energy": Energies,
    "dft_total_force": Forces,
    "scf_dipole": DipoleMomentPerSystem,
    "scf_quadrupole": QuadrupoleMomentPerSystem,
    "mulliken_partial_charges": PartialCharges,
    "lowdin_partial_charges": PartialCharges,
    "spin_multiplicity_per_atom": SpinMultiplicitiesPerAtom,
}

ase_neutral_sm1 = {
    "H": -1578.295303552632 * unit.kilojoule_per_mole,
    "C": -100080.45229773567 * unit.kilojoule_per_mole,
    "N": -143802.96775951583 * unit.kilojoule_per_mole,
    "O": -197590.0362883943 * unit.kilojoule_per_mole,
    "F": -262240.54180378723 * unit.kilojoule_per_mole,
    "P": -896392.2086813415 * unit.kilojoule_per_mole,
    "S": -1045618.4654004136 * unit.kilojoule_per_mole,
    "Cl": -1208456.0574764046 * unit.kilojoule_per_mole,
    "Fe": -3318703.03351438 * unit.kilojoule_per_mole,
    "Cu": -4308057.787521776 * unit.kilojoule_per_mole,
    "Zn": -4672738.253201201 * unit.kilojoule_per_mole,
    "Br": -6759223.343061076 * unit.kilojoule_per_mole,
    "Pd": -336331.45508597355 * unit.kilojoule_per_mole,
}

ase_neutral_sm3 = {
    "H": -1576.9155477694462 * unit.kilojoule_per_mole,
    "C": -100083.36715194868 * unit.kilojoule_per_mole,
    "N": -143809.0409112561 * unit.kilojoule_per_mole,
    "O": -197590.540959788 * unit.kilojoule_per_mole,
    "F": -262239.58335656545 * unit.kilojoule_per_mole,
    "P": -896376.398925346 * unit.kilojoule_per_mole,
    "S": -1045628.1497043676 * unit.kilojoule_per_mole,
    "Cl": -1208460.256209128 * unit.kilojoule_per_mole,
    "Fe": -3318523.879468734 * unit.kilojoule_per_mole,
    "Cu": -4307779.021706155 * unit.kilojoule_per_mole,
    "Zn": -4672408.576420245 * unit.kilojoule_per_mole,
    "Br": -6759227.058994985 * unit.kilojoule_per_mole,
    "Pd": -336067.11123574385 * unit.kilojoule_per_mole,
}


ase_neutral_sm5 = {
    "H": -1575.0660243748973 * unit.kilojoule_per_mole,
    "C": -100087.91787513118 * unit.kilojoule_per_mole,
    "N": -143816.78776389174 * unit.kilojoule_per_mole,
    "O": -197590.81177786744 * unit.kilojoule_per_mole,
    "F": -262238.8966632229 * unit.kilojoule_per_mole,
    "P": -896358.0665881358 * unit.kilojoule_per_mole,
    "S": -1045638.4717598899 * unit.kilojoule_per_mole,
    "Cl": -1208465.0562025441 * unit.kilojoule_per_mole,
    "Fe": -3318291.285941741 * unit.kilojoule_per_mole,
    "Cu": -4307362.891890421 * unit.kilojoule_per_mole,
    "Zn": -4671987.007626464 * unit.kilojoule_per_mole,
    "Br": -6759232.810209598 * unit.kilojoule_per_mole,
    "Pd": -335679.05159894767 * unit.kilojoule_per_mole,
}

ase_minus1_sm1 = {
    "H": -1576.2692981825426 * unit.kilojoule_per_mole,
    "C": -100082.74277133205 * unit.kilojoule_per_mole,
    "N": -143795.25559765278 * unit.kilojoule_per_mole,
    "O": -197601.1034560722 * unit.kilojoule_per_mole,
    "F": -262251.55831925524 * unit.kilojoule_per_mole,
    "P": -896388.9537269652 * unit.kilojoule_per_mole,
    "S": -1045636.9833501305 * unit.kilojoule_per_mole,
    "Cl": -1208506.5484449547 * unit.kilojoule_per_mole,
    "Fe": -3318603.9166225595 * unit.kilojoule_per_mole,
    "Cu": -4307840.322642327 * unit.kilojoule_per_mole,
    "Zn": -4672652.7744230535 * unit.kilojoule_per_mole,
    "Br": -6759277.915812828 * unit.kilojoule_per_mole,
    "Pd": -336214.7529675992 * unit.kilojoule_per_mole,
}

ase_minus1_sm3 = {
    "H": -1574.5068290400327*unit.kilojoule_per_mole,
    "C": -100085.523956817*unit.kilojoule_per_mole,
    "N": -143792.90525215017*unit.kilojoule_per_mole,
    "O": -197597.90633338375*unit.kilojoule_per_mole,
    "F": -262248.1290781242*unit.kilojoule_per_mole,
    "P": -896368.1518167559*unit.kilojoule_per_mole,
    "S": -1045649.9865841753*unit.kilojoule_per_mole,
    "Cl": -1208500.5691768017*unit.kilojoule_per_mole,
    "Fe": -3318509.996119875*unit.kilojoule_per_mole,
    "Cu": -4307626.651891471*unit.kilojoule_per_mole,
    "Zn": -4672403.6950625125*unit.kilojoule_per_mole,
    "Br": -6759277.193979238*unit.kilojoule_per_mole,
    "Pd": -336042.213154398*unit.kilojoule_per_mole,
}

ase_minus1_sm5 = {
    "H": -1571.2965950843613 * unit.kilojoule_per_mole,
    "C": -100097.90300624218 * unit.kilojoule_per_mole,
    "N": -143826.81031030868 * unit.kilojoule_per_mole,
    "O": -197622.62677242927 * unit.kilojoule_per_mole,
    "F": -262237.4182014358 * unit.kilojoule_per_mole,
    "P": -896426.8260784986 * unit.kilojoule_per_mole,
    "S": -1045659.2102843799 * unit.kilojoule_per_mole,
    "Cl": -1208499.9689096254 * unit.kilojoule_per_mole,
    "Fe": -3318331.5989786675 * unit.kilojoule_per_mole,
    "Cu": -4307449.384198708 * unit.kilojoule_per_mole,
    "Zn": -4672013.157154923 * unit.kilojoule_per_mole,
    "Br": -6759290.346818929 * unit.kilojoule_per_mole,
    "Pd": -335687.6818663071 * unit.kilojoule_per_mole,
}

ase_plus1_sm1 = {
    "H": -1578.383151626864 * unit.kilojoule_per_mole,
    "C": -100081.71631926112 * unit.kilojoule_per_mole,
    "N": -143808.40591272127 * unit.kilojoule_per_mole,
    "O": -197563.7206421346 * unit.kilojoule_per_mole,
    "F": -262243.89982271206 * unit.kilojoule_per_mole,
    "P": -896396.2582964546 * unit.kilojoule_per_mole,
    "S": -1045613.3736305279 * unit.kilojoule_per_mole,
    "Cl": -1208421.8268427218 * unit.kilojoule_per_mole,
    "Fe": -3318180.9843102996 * unit.kilojoule_per_mole,
    "Cu": -4307518.3251840025 * unit.kilojoule_per_mole,
    "Zn": -4672204.975124 * unit.kilojoule_per_mole,
    "Br": -6759206.748314817 * unit.kilojoule_per_mole,
    "Pd": -335774.7751280239 * unit.kilojoule_per_mole,
}

ase_plus1_sm3 = {
    "H": -1577.2675982874325 * unit.kilojoule_per_mole,
    "C": -100084.29663427873 * unit.kilojoule_per_mole,
    "N": -143816.56929997366 * unit.kilojoule_per_mole,
    "O": -197574.76279026177 * unit.kilojoule_per_mole,
    "F": -262243.28007424035 * unit.kilojoule_per_mole,
    "P": -896383.7136054877 * unit.kilojoule_per_mole,
    "S": -1045630.7307753554 * unit.kilojoule_per_mole,
    "Cl": -1208448.3221278673 * unit.kilojoule_per_mole,
    "Fe": -3317956.474122912 * unit.kilojoule_per_mole,
    "Cu": -4307189.9861974465 * unit.kilojoule_per_mole,
    "Zn": -4671793.16796686 * unit.kilojoule_per_mole,
    "Br": -6759223.7472838545 * unit.kilojoule_per_mole,
    "Pd": -335461.9676655856 * unit.kilojoule_per_mole,
}

ase_plus1_sm5 = {
    "H": -1574.5876344056426 * unit.kilojoule_per_mole,
    "C": -100089.1889216816 * unit.kilojoule_per_mole,
    "N": -143827.37546963338 * unit.kilojoule_per_mole,
    "O": -197579.17481110655 * unit.kilojoule_per_mole,
    "F": -262243.6705996112 * unit.kilojoule_per_mole,
    "P": -896370.6610044435 * unit.kilojoule_per_mole,
    "S": -1045645.1982444292 * unit.kilojoule_per_mole,
    "Cl": -1208459.2666888614 * unit.kilojoule_per_mole,
    "Fe": -3317708.2013844815 * unit.kilojoule_per_mole,
    "Cu": -4306751.717626212 * unit.kilojoule_per_mole,
    "Zn": -4671316.168908178 * unit.kilojoule_per_mole,
    "Br": -6759227.329785913 * unit.kilojoule_per_mole,
    "Pd": -335059.1960925312 * unit.kilojoule_per_mole,
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


def identify_tm_center(record: Record) -> int:
    """Identify the transition metal center in a record based on atomic numbers.

    Args:
        record (Record): The record to analyze.

    Returns:
        type: int
            The atomic number of the transition metal center

    """

    elements_of_interest = [46, 30, 26, 29]

    for element in elements_of_interest:
        atomic_numbers = record.get_property("atomic_numbers").value
        if element in atomic_numbers:
            return element


def get_system_breakdown(dataset):
    counts = {}
    for el in [46, 30, 26, 29]:
        counts[el] = 0
    for record_name in dataset.keys():
        record = dataset.get_record(record_name)
        tm_center = identify_tm_center(record)
        counts[tm_center] += 1
    return counts


tmqm_openff_dataset = create_dataset_from_hdf5(
    hdf5_filename="/home/cri/mf_datasets/hdf5_files/tmqm_openff_dataset/tmqm_openff_dataset_v1.3.hdf5",
    dataset_name="tmqm_openff_dataset",
    property_map=property_map,
)


for record_name in tmqm_openff_dataset.keys():
    record = tmqm_openff_dataset.get_record(record_name)

    total_charge = record.get_property("total_charge").value
    atomic_numbers = record.get_property("atomic_numbers").value
    spin_multiplicity = record.get_property("per_system_spin_multiplicity").value

    charge_state = ""
    spin_multiplicity_state = ""

    # determine charge state
    if np.all(total_charge == 0):
        charge_state = "neutral"
    elif np.all(total_charge == 1):
        charge_state = "plus1"
    elif np.all(total_charge == -1):
        charge_state = "minus1"

    # in general, we have spin multiplicities of 1, 3, or 5 in each configuration
    # so we will calculate the reference energy for a given charge state and all spin multiplicities

    # not the most elegant way to do this but it works fine
    if charge_state == "neutral":
        reference_energy_sm1 = calculate_reference_energy(
            atomic_numbers, ase_neutral_sm1
        )
        reference_energy_sm3 = calculate_reference_energy(
            atomic_numbers, ase_neutral_sm3
        )
        reference_energy_sm5 = calculate_reference_energy(
            atomic_numbers, ase_neutral_sm5
        )
    elif charge_state == "plus1":
        reference_energy_sm1 = calculate_reference_energy(atomic_numbers, ase_plus1_sm1)
        reference_energy_sm3 = calculate_reference_energy(atomic_numbers, ase_plus1_sm3)
        reference_energy_sm5 = calculate_reference_energy(atomic_numbers, ase_plus1_sm5)
    elif charge_state == "minus1":
        reference_energy_sm1 = calculate_reference_energy(
            atomic_numbers, ase_minus1_sm1
        )
        reference_energy_sm3 = calculate_reference_energy(
            atomic_numbers, ase_minus1_sm3
        )
        reference_energy_sm5 = calculate_reference_energy(
            atomic_numbers, ase_minus1_sm5
        )

    # each record may have multiple configurations with different spin multiplicities
    # so we need to create an array of reference energies that matches the shape of the spin multiplicity
    reference_energy = (
        np.zeros_like(spin_multiplicity, dtype=float) * unit.kilojoule_per_mole
    )
    for i, sm in enumerate(spin_multiplicity):
        if sm == 1:
            reference_energy[i] = reference_energy_sm1
        elif sm == 3:
            reference_energy[i] = reference_energy_sm3
        elif sm == 5:
            reference_energy[i] = reference_energy_sm5

    dft_total_energy = record.get_property("dft_total_energy")
    corrected_energy = (
        dft_total_energy.value * dft_total_energy.units - reference_energy
    )
    dft_total_energy_corrected = Energies(
        name="dft_total_energy_corrected",
        value=corrected_energy.m,
        units=corrected_energy.units,
    )
    record.add_property(dft_total_energy_corrected)

    tmqm_openff_dataset.update_record(record)

# write the whole dataset out to disk first

tmqm_openff_dataset.to_hdf5(
    file_path="/home/cri/mf_datasets/hdf5_files/tmqm_openff_dataset/v1.3_subsets/",
    file_name=f"tmqm_openff_full_dataset_corrected_v1.3c.hdf5",
)
seed = 42
# create a random seed for reproducibility
np.random.seed(seed)

### First, we will consider the full dataset and just remove a fixed 10% for testing

# get the total number of records and create an array of indices
n_records = tmqm_openff_dataset.total_records()
indices = np.arange(n_records)

# shuffle the indices in place
np.random.shuffle(indices)

# Let us just take the first 10% of the shuffled indices for our fixed test set
first_10 = int(n_records * 0.1)

# get the record names; we'll grab the first 10% of these
record_names = tmqm_openff_dataset.keys()

# create a new dataset with the first 10% of the shuffled indices
# we will then remove these records from the original dataset

logger.info("Creating TMQM fixed test subset...")
test_record_names = [record_names[i] for i in indices[:first_10]]

test_subset = SourceDataset(name="tmqm_openff_fixed_test_subset")

# let us loop over the test_record_names
for record_name in test_record_names:

    # fetch the record
    record = tmqm_openff_dataset.get_record(record_name)

    # add to the new subset
    test_subset.add_record(record)

    # remove from the original dataset
    tmqm_openff_dataset.remove_record(record_name)

new_dataset_neutral = SourceDataset(name="tmqm_openff_neutral_only")
new_dataset_charged = SourceDataset(name="tmqm_openff_charged_only")
new_dataset_plus1 = SourceDataset(name="tmqm_openff_plus1")
new_dataset_minus1 = SourceDataset(name="tmqm_openff_minus1")


# let us grab just the neutral molecules from the train/val set
for record_name in tmqm_openff_dataset.keys():
    record = tmqm_openff_dataset.get_record(record_name)
    if np.all(record.get_property("total_charge").value == 0):
        new_dataset_neutral.add_record(record)
    elif np.all(record.get_property("total_charge").value == 1):
        new_dataset_plus1.add_record(record)
        new_dataset_charged.add_record(record)
    elif np.all(record.get_property("total_charge").value == -1):
        new_dataset_minus1.add_record(record)
        new_dataset_charged.add_record(record)
    else:
        print(record_name, record.get_property("total_charge").value)


# Print out some info on each dataset
print(f"Total records in original dataset: {n_records}")
print(f"Total records in test subset: {test_subset.total_records()}")
print(f"Total records in training/val dataset: {tmqm_openff_dataset.total_records()}")
print(
    f"Total records in training/val dataset (neutral only): {new_dataset_neutral.total_records()}"
)
print(
    f"Total records in training/val dataset (charged only): {new_dataset_charged.total_records()}"
)
print(
    f"Total records in training/val dataset (+1 charged only): {new_dataset_plus1.total_records()}"
)
print(
    f"Total records in training/val dataset (-1 charged only): {new_dataset_minus1.total_records()}"
)

# print the number of configurations in each dataset
print(f"Total configurations in test subset: {test_subset.total_configs()}")

print(
    f"Total configurations in training/val dataset: {tmqm_openff_dataset.total_configs()}"
)
print(
    f"Total configurations in training/val dataset (neutral only): {new_dataset_neutral.total_configs()}"
)
print(
    f"Total configurations in training/val dataset (charged only): {new_dataset_charged.total_configs()}"
)
print(
    f"Total configurations in training/val dataset (+1 charged only): {new_dataset_plus1.total_configs()}"
)
print(
    f"Total configurations in training/val dataset (-1 charged only): {new_dataset_minus1.total_configs()}"
)


# let us save these to file:
test_subset.to_hdf5(
    file_path="/home/cri/mf_datasets/hdf5_files/tmqm_openff_dataset/v1.3_subsets/",
    file_name=f"tmqm_openff_full_dataset_test_seed_{seed}_v1.3c.hdf5",
)

tmqm_openff_dataset.to_hdf5(
    file_path="/home/cri/mf_datasets/hdf5_files/tmqm_openff_dataset/v1.3_subsets/",
    file_name=f"tmqm_openff_full_dataset_train_val_seed_{seed}_v1.3c.hdf5",
)

new_dataset_neutral.to_hdf5(
    file_path="/home/cri/mf_datasets/hdf5_files/tmqm_openff_dataset/v1.3_subsets/",
    file_name=f"tmqm_openff_full_dataset_train_val_neutral_seed_{seed}_v1.3c.hdf5",
)
new_dataset_charged.to_hdf5(
    file_path="/home/cri/mf_datasets/hdf5_files/tmqm_openff_dataset/v1.3_subsets/",
    file_name=f"tmqm_openff_full_dataset_train_val_charged_seed_{seed}_v1.3c.hdf5",
)
new_dataset_plus1.to_hdf5(
    file_path="/home/cri/mf_datasets/hdf5_files/tmqm_openff_dataset/v1.3_subsets/",
    file_name=f"tmqm_openff_full_dataset_train_val_plus1_seed_{seed}_v1.3c.hdf5",
)
new_dataset_minus1.to_hdf5(
    file_path="/home/cri/mf_datasets/hdf5_files/tmqm_openff_dataset/v1.3_subsets/",
    file_name=f"tmqm_openff_full_dataset_train_val_minus1_seed_{seed}_v1.3c.hdf5",
)


def plot_energy_distribution(dataset, title, filename):
    import matplotlib.pyplot as plt

    energies = []
    for record_name in dataset.keys():
        record = dataset.get_record(record_name)
        energy_prop = record.get_property("dft_total_energy_corrected")
        energy = energy_prop.value * energy_prop.units
        # atomic_numbers = record.get_property("atomic_numbers").value
        # reference_energy = calculate_reference_energy(atomic_numbers, ase)
        energies.append(energy.m_as(unit.kilojoule_per_mole))

    plt.figure(figsize=(8, 6))
    plt.hist(np.concatenate(energies).reshape(-1), bins=100, color="blue", alpha=0.7)
    plt.title(title)
    plt.xlabel("Energy (kJ/mol)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


plot_energy_distribution(
    dataset=tmqm_openff_dataset,
    title="Full dataset",
    filename="/home/cri/mf_datasets/hdf5_files/tmqm_openff_dataset/v1.3_subsets/tmqm_openff_full_dataset_energy_distribution_c.png",
)
plot_energy_distribution(
    dataset=new_dataset_neutral,
    title="Neutral molecules only",
    filename="/home/cri/mf_datasets/hdf5_files/tmqm_openff_dataset/v1.3_subsets/tmqm_openff_neutral_only_energy_distribution_c.png",
)
plot_energy_distribution(
    dataset=new_dataset_charged,
    title="Charged molecules only",
    filename="/home/cri/mf_datasets/hdf5_files/tmqm_openff_dataset/v1.3_subsets/tmqm_openff_charged_only_energy_distribution_c.png",
)
plot_energy_distribution(
    dataset=test_subset,
    title="Test subset (all molecules)",
    filename="/home/cri/mf_datasets/hdf5_files/tmqm_openff_dataset/v1.3_subsets/tmqm_openff_test_subset_energy_distribution_c.png",
)
