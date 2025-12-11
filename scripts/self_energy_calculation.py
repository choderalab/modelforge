from modelforge.dataset.dataset import initialize_datamodule
from loguru import logger

# This script will perform the self-energy calculation for multiple datasets

datasets = {
    # "qm9": {
    #     "properties_of_interest": [
    #         "positions",
    #         "internal_energy_at_0K",
    #         "atomic_numbers",
    #     ],
    #     "properties_assignment": {
    #         "positions": "positions",
    #         "E": "internal_energy_at_0K",
    #         "atomic_numbers": "atomic_numbers",
    #     },
    # },
    # "ani1x": {
    #     "properties_of_interest": ["positions", "wb97x_dz_energy", "atomic_numbers"],
    #     "properties_assignment": {
    #         "positions": "positions",
    #         "E": "wb97x_dz_energy",
    #         "atomic_numbers": "atomic_numbers",
    #     },
    # },
    # "ani2x": {
    #     "properties_of_interest": ["positions", "energies", "atomic_numbers"],
    #     "properties_assignment": {
    #         "positions": "positions",
    #         "E": "energies",
    #         "atomic_numbers": "atomic_numbers",
    #     },
    # },
    # "spice1": {
    #     "properties_of_interest": ["positions", "dft_total_energy", "atomic_numbers"],
    #     "properties_assignment": {
    #         "positions": "positions",
    #         "E": "dft_total_energy",
    #         "atomic_numbers": "atomic_numbers",
    #     },
    # },
    # "spice2": {
    #     "properties_of_interest": ["positions", "dft_total_energy", "atomic_numbers"],
    #     "properties_assignment": {
    #         "positions": "positions",
    #         "E": "dft_total_energy",
    #         "atomic_numbers": "atomic_numbers",
    #     },
    # },
    # "spice1_openff": {
    #     "properties_of_interest": ["positions", "dft_total_energy", "atomic_numbers"],
    #     "properties_assignment": {
    #         "positions": "positions",
    #         "E": "dft_total_energy",
    #         "atomic_numbers": "atomic_numbers",
    #     },
    # },
    "spice2_openff": {
        "properties_of_interest": ["positions", "dft_total_energy", "atomic_numbers"],
        "properties_assignment": {
            "positions": "positions",
            "E": "dft_total_energy",
            "atomic_numbers": "atomic_numbers",
        },
    },
    # "phalkethoh": {
    #     "properties_of_interest": ["positions", "dft_total_energy", "atomic_numbers"],
    #     "properties_assignment": {
    #         "positions": "positions",
    #         "E": "dft_total_energy",
    #         "atomic_numbers": "atomic_numbers",
    #     },
    # },
}

for dataset_name in datasets:
    logger.info(f"Processing dataset: {dataset_name}")

    properties_of_interest = datasets[dataset_name]["properties_of_interest"]
    properties_assignment = datasets[dataset_name]["properties_assignment"]
    dm = initialize_datamodule(
        dataset_name=dataset_name,
        batch_size=200000,
        regression_ase=False,
        remove_self_energies=False,
        version_select="latest",
        local_cache_dir="./local_cache",
        dataset_cache_dir="~/dataset_cache",
        properties_of_interest=properties_of_interest,
        properties_assignment=properties_assignment,
    )

    regressed = dm.calculate_self_energies(dm.torch_dataset)

    # write out the results
    # write out both as a dictionary and yaml formatted
    print(f"Self-energies for dataset: {dataset_name}")
    for key, value in regressed.items():
        print(f'"{key}": {value.m}*unit.{value.u},')

    # write out a yaml formatted version to file
    file_name = f"self_energies_{dataset_name}.yaml"
    logger.info(f"Writing self-energies to: {file_name}")
    with open(file_name, "w") as f:
        for key, value in regressed.items():
            f.write(f"{key}: {value.m}*{value.u},")
