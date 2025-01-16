import h5py

# This file will read in the tmqm hdf5 file and create a few datasets of restricted configurations
# Functionality is being added to filter datasets at run time, but this is designed to be an intermediate solution.

file_path = "/home/cri/datasets/hdf5_files/tmqm_dataset_v0.hdf5"


# Create a dataset that contains transition metals Pd, Zn, Fe, Cu
# This should be 27685 molecules
primary_tm_to_extract = [46, 30, 26, 29]
primary_tm_to_extract_list = []

# a second set will include Pd, Zn, Fe, Cu, Ni, Pt, Ir, Rh, Cr, Ag
# This will be 60637 molecules
primary_and_secondary_tm_to_extract = [46, 30, 26, 29, 28, 78, 77, 45, 24, 47]
primary_and_secondary_tm_to_extract_list = []

# Limit organics to C, H, P, S O, N, F Cl, Br by excluding
# any that contain B, Si, As, Se, I

exclude_organics = [5, 14, 33, 16, 53]
primary_tm_excluded_organics_list = []
primary_and_secondary_tm_excluded_organics_list = []

file_output_primary = "/home/cri/datasets/hdf5_files/tmqm_dataset_PdZnFeCu_v0.hdf5"
file_output_primary_excluded_organics = (
    "/home/cri/datasets/hdf5_files/tmqm_dataset_PdZnFeCu_CHPSONFClBr_v0.hdf5"
)
file_output_primary_secondary = (
    "/home/cri/datasets/hdf5_files/tmqm_dataset_PdZnFeCuNiPtIrRhCrAg_v0.hdf5"
)
file_output_primary_secondary_excluded_organics = "/home/cri/datasets/hdf5_files/tmqm_dataset_PdZnFeCuNiPtIrRhCrAg_CHPSONFClBr_v0.hdf5"


with h5py.File(file_path, "r") as f:
    keys = list(f.keys())
    for i in range(len(keys)):
        key = keys[i]
        atomic_numbers = f[key]["atomic_numbers"][()]

        for tm_atomic_number in primary_tm_to_extract:
            if tm_atomic_number in atomic_numbers:
                primary_tm_to_extract_list.append(key)

                add_mol = True
                for organic_atomic_number in exclude_organics:
                    if organic_atomic_number in atomic_numbers:
                        add_mol = False
                if add_mol:
                    primary_tm_excluded_organics_list.append(key)

        for tm_atomic_number in primary_and_secondary_tm_to_extract:
            if tm_atomic_number in atomic_numbers:
                primary_and_secondary_tm_to_extract_list.append(key)

                add_mol = True
                for organic_atomic_number in exclude_organics:
                    if organic_atomic_number in atomic_numbers:
                        add_mol = False
                if add_mol:
                    primary_and_secondary_tm_excluded_organics_list.append(key)

with h5py.File(file_output_primary, "w") as fout:
    with h5py.File(file_path, "r") as fin:
        for key in primary_tm_to_extract_list:
            fin.copy(fin[key], fout, key)

with h5py.File(file_output_primary_excluded_organics, "w") as fout:
    with h5py.File(file_path, "r") as fin:
        for key in primary_tm_excluded_organics_list:
            fin.copy(fin[key], fout, key)

with h5py.File(file_output_primary_secondary, "w") as fout:
    with h5py.File(file_path, "r") as fin:
        for key in primary_and_secondary_tm_to_extract_list:
            fin.copy(fin[key], fout, key)

with h5py.File(file_output_primary_secondary_excluded_organics, "w") as fout:
    with h5py.File(file_path, "r") as fin:
        for key in primary_and_secondary_tm_excluded_organics_list:
            fin.copy(fin[key], fout, key)

print("number of molecules in primary set: ", len(primary_tm_to_extract_list))
print(
    "number of molecules in primary set excluding organics: ",
    len(primary_tm_excluded_organics_list),
)
print(
    "number of molecules in primary and secondary set: ",
    len(primary_and_secondary_tm_to_extract_list),
)
print(
    "number of molecules in primary and secondary set excluding organics: ",
    len(primary_and_secondary_tm_excluded_organics_list),
)
