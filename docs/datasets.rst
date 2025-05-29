Datasets
===============

The dataset module in modelforge provides a  suite of functions and classes designed to retrieve and transform quantum mechanics (QM) datasets into a format compatible with `torch.utils.data.Dataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`_, facilitating the training of machine learning potentials. The module supports actions related to data storage, caching, retrieval, and the conversion of stored modelforge curated HDF5 files into PyTorch-compatible datasets for training purposes.

Modelforge currently provides a host of datasets containing a variety of molecular structures and properties. These datasets are curated into HDF5 formated files designed to be compatible with modelforge and hosted on zenodo.org (see the `zenodo modelforge community <https://zenodo.org/communities/modelforge/records>`_), making them easily accessible for training machine learning models. These datasets can easily specified and all the relevant files downloaded and cached using the :class:`~modelforge.dataset.HDF5Dataset` class. In all cases, a 1000 configuration test dataset is provided for each dataset; several datasets also provide various subsets (e.g., limited to a subset of elements). The specific version can be specified; version information is embedded in .yaml files in the dataset directory (`~modelforge/dataset/yaml_files`).

Local datasets can also be used, assuming they are stored in the modelforge compatible HDF5 format.  These can be specified by providing a path to a yaml file that contains the dataset metadata in the dataset configuration file.


The following datasets are available as part of `modelforge` (dataset names used to specify the dataset in modelforge are provided in parentheses); additional details of each dataset, including the corresponding yaml files (which provide a list of available versions of each dataset) are provided at the end of this page:

- **QM9 (qm9)**
- **ANI1x (ani1x)**
- **ANI2X (ani2x)**
- **SPICE 1 (spice1)**
- **SPICE 2 (spice2)**
- **SPICE 1 OpenFF (spice1_openff)**
- **SPICE 2 OpenFF (spice2_openff)**
- **Fe II (fe_ii)**
- **tmQM (tmqm)**
- **tmQM-xtb (tmqm_xtb)**
- **PhAlkEthOH (PhAlkEthOH)**


Postprocessing of dataset entries
-----------------------------------

Two common postprocessing operations are performed for training machine learned potentials:

- *Removing Self-Energies*: Self-energies are per-element offsets added to the
  total energy of a system. These offsets are not useful for training
  machine-learned potentials and can be removed to provide cleaner training
  data.
- *Normalization and Scaling*: Normalize the energies and other properties to
  ensure they are on a comparable scale, which can improve the stability and
  performance of the machine learning model. Note that this is done when atomic energies are predicted, i.e. the atomic energy (`E_i`) is scaled using the atomic energy distribution obtained from the training dataset: `E_i = E_i_stddev * E_i_pred + E_i_mean`.


Interacting with the Dataset Module
-----------------------------------

The dataset module provides a :class:`~modelforge.dataset.DataModule` class for
preparing and setting up datasets for training. Designed to integrate seamlessly
with PyTorch Lightning, the :class:`~modelforge.dataset.DataModule` class
provides a user-friendly interface for dataset preparation and loading.

The following example demonstrates how to use the :class:`~modelforge.dataset.DataModule` class to prepare and set up a dataset for training:


.. code-block:: python

    from modelforge.dataset import DataModule
    from modelforge.dataset.utils import RandomRecordSplittingStrategy

    dataset_name = "QM9"
    splitting_strategy = RandomRecordSplittingStrategy() # split randomly on system level
    batch_size = 64
    version_select = "latest" 
    remove_self_energies = True # remove the atomic self energies
    regression_ase = False      # use the atomic self energies provided by the dataset

    data_module = DataModule(
        name=dataset_name,
        properties_of_interest=["atomic_numbers", "positions", "internal_energy_at_0K"]
        properties_assignment={
            "E": "energy",
            "atomic_numbers": "atomic_numbers",
            "positions": "positions",
        },
        splitting_strategy=splitting_strategy,
        batch_size=batch_size,
        version_select=version_select,
        remove_self_energies=remove_self_energies,
        regression_ase=regression_ase,
        local_cache_dir="~/modelforge_run",
        dataset_cache_dir="~/modelforge_hdf5_files",
    )

    # Prepare the data (downloads, processes, and caches if necessary)
    data_module.prepare_data()

    # Setup the data for training, validation, and testing
    data_module.setup()

.. _dataset-configuration:

`version_select` specifies the version of the dataset to use and `properties_of_interest` specifes the keys within the dataset to read; a full list of version and available properties for each version can be found in the dataset metadata yaml file (e.g., `qm9.yaml` for the QM9 dataset). The `properties_assignment` dictionary maps these properties to the variables used internally as part of the :class:`~modelforge.utils.prop.PropertyNames` class.

.. code-block:: python

    class PropertyNames:
        atomic_numbers: str
        positions: str  # Positions
        E: str  # Energy
        F: Optional[str] = None  # Forces
        total_charge: Optional[str] = None  # Total charge
        dipole_moment: Optional[str] = None  # Dipole moment
        S: Optional[str] = None  # Spin multiplicity,

The `splitting_strategy` determines how the dataset is split into training, validation, and test sets; in this case, it uses :class:`modelforge.dataset.utils.RandomRecordSplittingStrategy`, where splitting operates on the unique systems/records (i.e., keeping configurations of the same system together). Other splitting strategies include :class:`modelforge.dataset.utils.RandomSplittingStrategy` which splits the dataset randomly across all configurations (i.e., not group configurations of the same system together), and :class:`modelforge.dataset.utils.FirstComeFirstServeSplittingStrategy` which splits the configurations in the order they are encountered in the dataset. The `batch_size` specifies the number of configurations to include in each batch during training. The `remove_self_energies` flag indicates whether to remove self-energies from the dataset (i.e. summing up the total atomic self energies and subtracting from the system). The `regression_ase` flag indicates whether to use the atomic self energies provided (in the metadata yaml file, if available) or to calculate them via regression. `local_cache_dir` specifies the directory where files generated during training will be saved.  `dataset_cache_dir` specifies the directory where the dataset files will be cached.  The reason to separate local_cache_dir and dataset_cache_dir is that it enables hdf5 files can be shared in a central location across multiple training runs (saving storage space and avoiding additional time spent downloading files), while allowing each run to have its own local cache directory for temporary files and results.

Dataset Configuration
------------------------------------

Dataset input configuration is typically managed using a TOML file. This configuration file is crucial during the training process as it provides values that need to be specified  for the :class:`~modelforge.dataset.DataModule` class, ensuring a flexible and customizable setup.

Below is a minimal example of a dataset configuration for the QM9 dataset.

.. literalinclude:: ../modelforge/tests/data/dataset_defaults/qm9.toml
   :language: toml
   :caption: QM9 Dataset Configuration

.. warning::
    The ``version_select`` field in the example indicates the use of a small subset of the QM9 dataset. To utilize the full dataset, set this variable to ``latest``.


Explanation of fields in `qm9.toml`:

- `dataset_name`: Specifies the name of the dataset. For this example, it is QM9.
- `number_of_worker`: Determines the number of worker threads for data loading. Increasing the number of workers can speed up data loading but requires more memory.
- `version_select`: Indicates the version of the dataset to use. In this example, it points to a small subset of the dataset for quick testing. To use the full QM9 dataset, set this variable to `latest`.
- `properties_of_interest`: Lists the properties of interest to load from the hdf5 file.
- `properties_assignment`: Maps the properties of interest to the corresponding fields in the dataset. This mapping is crucial for the correct loading of properties during training; note, many datasets contain multiple properties can potentially be swapped (e.g., energy calculated with or without dispersion corrections, different charge population schemes, different levels of theory, etc.).  Any properties listed here must appear in the properties of interest list; the code will raise a validation error if this condition is not met.
- `element_filter`: A filter to select systems with or without certain elements, which are denoted by atomic numbers. If a positive number is provided, then a datapoint that includes that element will be included. A negative values indicates which elements to exclude. For example, [[29]], selects all systems containing copper (29). [[29, -17]] selects all systems containing copper (29), but excludes from that list any that also contain chlorine (17). [[29, 1, -17]] would select all systems that contain copper (29) and hydrogen (H), and do not include chlorine (17). Everything contain within the same brackets acts as an "and" (i.e., all criteria must be satisfied). Providing two separate sublists acts as an "or". For example, [[29,1], [78,-17]], states that a molecule can either have [copper (29) and hydrogen (1)] OR [platinum (78) and not chlorine (17)]. Leaving this field as an empty list or remove it will disable this element filtering feature.

Note, to use a local dataset, you can specify the path to the dataset yaml file by adding `local_yaml_file` to the toml file and setting it to the path of the yaml file.

yaml Metadata File Structure
------------------------------------

The aforementioned datasets are accompanied by metadata files in YAML format, which provide essential information about the dataset, including its version, properties, and other relevant details, along with the downloard url used to fetch the dataset. These metadata files are stored in the `~modelforge/dataset/yaml_files` directory.

Below is a fictional example of a metadata to demonstrate the key fields which includes the dataset name, version, description, atomic self-energies, and available properties.

.. code-block:: yaml

    dataset: fictional_dataset_name
    latest: full_dataset_v1.1
    latest_test: nc_1000_v1.1

    description: "A description of the dataset."

    atomic_self_energies:
      H: -1400.0 * kilojoule_per_mole
      C: -10000.0 * kilojoule_per_mole

    full_dataset_v1.1:
      about: "This provides a curated hdf5 file for the fictional dataset designed to be compatible
        with modelforge. This dataset contains 1234 unique records for 123456 total
        configurations."
      hdf5_schema: 2
      available_properties:
      - atomic_numbers
      - positions
      - dft_energy
      remote_dataset:
        doi: 10.1234/fictional_dataset.v1.1
        url: https://zenodo.org/records/record_id/files/fictional_dataset_v1.1.hdf5.gz
        gz_data_file:
          file_name: fictional_dataset_v1.1.hdf5.gz
          length: 123456 # Length of the gzipped file in bytes
          md5: gzip_checksum_value
        hdf5_data_file:
          file_name: fictional_dataset_v1.1.hdf5
          md5: hdf5_checksum_value

To specify metadata for a local dataset, the `remote_dataset` field can be omitted and replaced with the field `local_dataset` as shown below:

.. code-block:: yaml
    dataset: fictional_dataset_name
    latest: full_dataset_v1.1
    latest_test: nc_1000_v1.1

    description: "A description of the dataset."

    atomic_self_energies:
      H: -1400.0 * kilojoule_per_mole
      C: -10000.0 * kilojoule_per_mole

    full_dataset_v1.1:
      about: "This provides a curated hdf5 file for the fictional dataset designed to be compatible
        with modelforge. This dataset contains 1234 unique records for 123456 total
        configurations."
      hdf5_schema: 2
      available_properties:
      - atomic_numbers
      - positions
      - dft_energy
      local_dataset:
        hdf5_data_file:
          file_name: path_to_file/local_fictional_dataset_v1.1_ntc_10.hdf5
          md5: hdf5_checksum_value


Available Datasets and Versions
----------------------------------

Below is a description of the available datasets and their corresponding metadata yaml files. These files can be found in the `~modelforge/dataset/yaml_files` directory, and they provide detailed information about each dataset, including its version, properties, and download URLs.  As previously mentioned, for each dataset, multiple versions may be available.  A 1000 configuration test dataset is provided for each dataset primarily useful for testing; several datasets also provide various subsets (e.g., limited to a subset of elements).

- **ANI1x (ani1x)**:  dataset includes ~5 million density function theory calculations for small organic molecules containing H, C, N, and O. A subset of ~500k are computed with accurate coupled cluster methods.

    ANI-1x dataset:
        Smith, J. S.; Nebgen, B.; Lubbers, N.; Isayev, O.; Roitberg, A. E. Less Is More: Sampling Chemical Space with Active Learning. J. Chem. Phys. 2018, 148 (24), 241733. https://doi.org/10.1063/1.5023802

    ANI-1ccx dataset:
        Smith, J. S.; Nebgen, B. T.; Zubatyuk, R.; Lubbers, N.; Devereux, C.; Barros, K.; Tretiak, S.; Isayev, O.; Roitberg, A. E. Approaching Coupled Cluster Accuracy with a General-Purpose Neural Network Potential through Transfer Learning. Nat. Commun. 2019, 10 (1), 2903. https://doi.org/10.1038/s41467-019-10827-4

    ωB97x/def2-TZVPP data:
        Zubatyuk, R.; Smith, J. S.; Leszczynski, J.; Isayev, O. Accurate and Transferable Multitask Prediction of Chemical Properties with an Atoms-in-Molecules Neural Network. Sci. Adv. 2019, 5 (8), eaav6490. https://doi.org/10.1126/sciadv.aav6490

.. literalinclude:: ../modelforge/dataset/yaml_files/ani1x.yaml
    :language: yaml
    :caption: ANI1x Dataset yaml Metadata

- **ANI2X (ani2x)**: The ANI-2x data set includes properties for small organic molecules that contain H, C, N, O, S, F, and Cl.  This dataset contains 9651712 conformers for nearly 200,000 molecules. This will fetch data generated with the ωB97X/631Gd level of theory used in the original ANI-2x paper, calculated using Gaussian 09.

    Devereux, C, Zubatyuk, R., Smith, J. et al. Extending the applicability of the ANI deep learning molecular potential to sulfur and halogens. Journal of Chemical Theory and Computation 16.7 (2020): 4192-4202. https://doi.org/10.1021/acs.jctc.0c00121

.. literalinclude:: ../modelforge/tdataset/yaml_files/ani2x.yaml
    :language: yaml
    :caption: ANI2x Dataset yaml Metadata

- **Fe II (fe_ii)**:  The Fe(II) dataset includes 28834 total configurations of 384 unique Fe(II) organometallic complexes. Specifically, this includes 15568 HS geometries and 13266 LS geometries. These complexes originate from the Cambridge Structural Database (CSD) as curated by Nandy, et al. (Journal of Physical Chemistry Letters (2023), 14 (25), 10.1021/acs.jpclett.3c01214), and were filtered into “computation-ready” complexes, (those where both oxidation states and charges are already specified without hydrogen atoms missing in the structures), following the procedure outlined by Arunachalam, et al. (Journal of Chemical Physics (2022), 157 (18), 10.1063/5.0125700).

    Hongni Jin and Kenneth M. Merz Jr, Modeling Fe(II) Complexes Using Neural Networks. Journal of Chemical Theory and Computation 2024 20 (6), 2551-2558 https://dx.doi.org/10.1021/acs.jctc.4c00063

.. literalinclude:: ../modelforge/dataset/yaml_files/fe_ii.yaml
    :language: yaml
    :caption: Fe II Dataset yaml Metadata

- **PhAlkEthOH (PhAlkEthOH)**: PhAlkEthOH: Phenyls, Alkanes, Ethers, and Alcohols (OH). The PhAlkEthOH dataset contains a collection of optimized trajectories of linear and cyclic molecules containing phyl rings, small alkanes, ethers, and alcohols containing only elements carbon, oxygen and hydrogen. For each unique system, configurations correspond to snapshots from the optimization trajectory. All QM datapoints were generated using B3LYP-D3BJ/DZVP level of theory, the default theory used for force field development by the Open Force Field Initiative.

    Bannan CC, Mobley D. ChemPer: An Open Source Tool for Automatically Generating SMIRKS Patterns. ChemRxiv. 2019; https://dx.doi.org/10.26434/chemrxiv.8304578.v1

    Wang Y, Fass J, Kaminow B, Herr JE, Rufa D, Zhang I, Pulido I, Henry M, Macdonald HE, Takaba K, Chodera JD. End-to-end differentiable construction of molecular mechanics force fields. Chemical Science. 2022;13(41):12016-33. https://dx.doi.org/10.1039/d2sc02739a

.. literalinclude:: ../modelforge/dataset/yaml_files/phalkethoh.yaml
    :language: yaml
    :caption: PhAlkEthOH Dataset yaml Metadata

**QM9 (qm9)**: A dataset of 134k small organic molecules, each containing up to 9 heavy atoms (C, O, N, F) and up to 29 atoms in total. It includes properties such as energies, forces, and dipole moments.

    Ramakrishnan, R., Dral, P., Rupp, M. et al. 'Quantum chemistry structures and properties of 134 kilo molecules.'Sci Data 1, 140022 (2014). https://doi.org/10.1038/sdata.2014.22

.. literalinclude:: ../modelforge/dataset/yaml_files/qm9.yaml
    :language: yaml
    :caption: QM9 Dataset yaml Metadata

- **SPICE 1 (spice1)**: The SPICE dataset contains 1.1 million conformations for a 19238 unique small molecules, dimers, dipeptides, and solvated amino acids. It includes 15 elements  (H, Li, C, N, O, F, Na, Mg, P, S, Cl, K, Ca, Br, I)., charged and uncharged molecules, and a wide range of covalent and non-covalent interactions. It provides both forces and energies calculated at the ωB97M-D3(BJ)/def2-TZVPPD level of theory, using Psi4 1.4.1 along with other useful quantities such as multipole moments and bond orders.

    Eastman, P., Behara, P.K., Dotson, D.L. et al. SPICE, A Dataset of Drug-like Molecules and Peptides for Training Machine Learning Potentials. Sci Data 10, 11 (2023). https://doi.org/10.1038/s41597-022-01882-6

.. literalinclude:: ../modelforge/dataset/yaml_files/spice1.yaml
    :language: yaml
    :caption: SPICE 1 Dataset yaml Metadata

- **SPICE 1 OpenFF (spice1_openff)**: The full SPICE 1 OpenFF dataset is a subset of the SPICE 1 dataset, and includes 18782 unique records for 1106949 total configurations for 14 different elements (H, Li, C, N, O, F, Na, Mg, P, S, Cl, K, Ca, Br). All QM datapoints were generated using B3LYP-D3BJ/DZVP level of theory as this is the default theory used for force field development by the Open Force Field Initiative.
  was generated using ωB97M-D3(BJ)/def2-TZVPPD level of theory.


.. literalinclude:: ../modelforge/dataset/yaml_files/spice1_openff.yaml
    :language: yaml
    :caption: SPICE 1 OpenFF Dataset yaml Metadata

- **SPICE 2 (spice2)**: The SPICE2 dataset contains conformations for a diverse set of small molecules, dimers, dipeptides, and solvated amino acids. It includes 17 elements (H, Li, B, C, N, O, F, Na, Mg, Si, P, S, Cl, K, Ca, Br, I), charged and uncharged molecules, and a wide range of covalent and non-covalent interactions. It provides both forces and energies calculated at the ωB97M-D3(BJ)/def2-TZVPPD level of theory, using Psi4 along with other useful quantities such as multipole moments and bond orders.

    Eastman, P., Pritchard, B. P., Chodera, J. D., & Markland, T. E. Nutmeg and SPICE: models and data for biomolecular machine learning. Journal of chemical theory and computation, 20(19), 8583-8593 (2024). https://doi.org/10.1021/acs.jctc.4c00794

.. literalinclude:: ../modelforge/dataset/yaml_files/spice2.yaml
    :language: yaml
    :caption: SPICE 2 Dataset yaml Metadata

- **SPICE 2 OpenFF (spice2_openff)**: The SPICE 2 OpenFF dataset is a subset of the SPICE 2 dataset, and includes 112628 unique records for 1971769 total configurations for 16 elements (H, Li, B, C, N, O, F, Na, Mg, Si, P, S, Cl, K, Ca, Br) in both charged and uncharged molecules, and a wide range of covalent and non-covalent interactions. All QM datapoints were generated using B3LYP-D3BJ/DZVP level of theory as this is the default theory used for force field development by the Open Force Field Initiative.

.. literalinclude:: ../modelforge/dataset/yaml_files/spice2_openff.yaml
    :language: yaml
    :caption: SPICE 2 OpenFF Dataset yaml Metadata

- **tmQM (tmqm)**: The tmQM dataset contains the geometries and properties of 108,541 mononuclear complexes extracted from the Cambridge Structural Database, including Werner, bioinorganic, and organometallic complexes based on a large variety of organic ligands and 30 transition metals (the 3d, 4d, and 5d from groups 3 to 12). All complexes are closed-shell, with a formal charge in the range {+1, 0, −1}e

    David Balcells and Bastian Bjerkem Skjelstad, tmQM Dataset—Quantum Geometries and Properties of 86k Transition Metal Complexes. Journal of Chemical Information and Modeling 2020 60 (12), 6135-6146 https://dx.doi.org/10.1021/acs.jcim.0c01041"

.. literalinclude:: ../modelforge/dataset/yaml_files/tmqm.yaml
    :language: yaml
    :caption: tmQM Dataset yaml Metadata

- **tmQM-xtb (tmqm_xtb)**: The tmQM-xtb dataset include configurations generated using GFN2-xTB-based MD simulations starting from the energy-minimized geometries in the tmQM dataset. Energies, forces, charges, and dipole moments were calculated using the GFN2-xTB method. Several variants of the dataset are available, generated using different temperatures for MD sampling.


.. literalinclude:: ../modelforge/dataset/yaml_files/tmqm_xtb.yaml
    :language: yaml
    :caption: tmQM-xtb Dataset yaml Metadata