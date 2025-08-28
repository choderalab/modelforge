Dataset Module
===============

The dataset module in modelforge provides a  suite of functions and classes designed to retrieve and transform quantum mechanics (QM) datasets into a format compatible with `torch.utils.data.Dataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`_ as well as Pytorch Lightning `LightningDataModule <https://lightning.ai/docs/pytorch/stable/data/datamodule.html#lightningdatamodule>`_,  facilitating the training of machine learning potentials. The module supports actions related to data storage, caching, retrieval, and the conversion of stored modelforge curated HDF5 files into PyTorch-compatible datasets for training purposes.

Modelforge currently provides a host of datasets containing a variety of molecular structures and properties. These datasets are curated into HDF5 formated files designed to be compatible with modelforge and hosted on zenodo.org (see the `zenodo modelforge community <https://zenodo.org/communities/modelforge/records>`_); the udnerlying :class:`~modelforge.dataset.HDF5Dataset` class provides a framework to download, cache, and process these files into a format compatible with `torch.utils.data.Dataset`, as previously noted. Local datasets can also be used that are stored in modelforge compatible HDF5 formats, allowing users to work with their own datasets without needing to upload them to a remote server or modifying the modelforge source.  These can be specified by providing a configuration file, as will be described below.


Dataset Configuration TOML file
------------------------------------

Dataset input configuration is typically managed using a TOML file. This configuration file is crucial during the training process as it provides values that need to be specified  for the :class:`~modelforge.dataset.DataModule` class, ensuring a flexible and customizable setup.

Below is a minimal example of a dataset configuration for the QM9 dataset.

.. literalinclude:: ../modelforge/tests/data/dataset_defaults/qm9.toml
   :language: toml
   :caption: QM9 Dataset Configuration

.. warning::
    The ``version_select`` field in the example indicates the use of a small subset of the QM9 dataset. To utilize the full dataset, set this variable to ``latest``.


Explanation of the possible fields in the dataset configuration file:

- `dataset_name`: Specifies the name of the dataset. For this example, it is QM9.
- `version_select`: Indicates the version of the dataset to use. In this example, it points to a small subset of the dataset for quick testing. To use the full QM9 dataset, set this variable to `latest`.
- `number_of_worker`: Determines the number of worker threads for data loading. Increasing the number of workers can speed up data loading but requires more memory. Must be 1 or greater.
- `pin_memory`: A boolean flag indicating whether to pin memory for faster data transfer to the GPU. This is useful when training on a GPU and can improve performance by reducing data transfer times. Defaults to `True`.
- `properties_of_interest`: Lists the properties of interest to load from the hdf5 file. This should include the properties that are relevant for training the model. The properties listed here must match those available in the dataset metadata; otherwise, a validation error will be raised. Loading properties that will not be used during training will use more memory.
- `properties_assignment`: Maps the properties of interest to the corresponding fields in the dataset. This mapping is crucial for the correct loading of properties during training; note, many datasets contain multiple properties can potentially be swapped (e.g., energy calculated with or without dispersion corrections, different charge population schemes, different levels of theory, etc.).  Any properties listed here must appear in the properties of interest list; the code will raise a validation error if this condition is not met.  The possible fields  to assign are defined by the :class:`~modelforge.utils.prop.PropertyNames`, which is listed below. Note, by default atomic_numbers, positions, and energy (E) are always required to be set.

.. code-block:: python

    class PropertyNames:
        atomic_numbers: str # per-atom atomic numbers (atomic numbers are integers)
        positions: str  # per-atom positions (cartesian coordinates)
        E: str  # per-system energy (total energy)
        F: Optional[str] = None  # per-atom forces
        total_charge: Optional[str] = None  # per-system total charge
        dipole_moment: Optional[str] = None  # per-system dipole moment
        spin_multiplicity: Optional[str] = None  # per-system spin multiplicity
        partial_charges: Optional[str] = None  # per-atom partial charges

- `element_filter`: A filter to select systems with or without certain elements, which are denoted by atomic numbers. If a positive number is provided, then a datapoint that includes that element will be included. A negative values indicates which elements to exclude. For example, [[29]], selects all systems containing copper (29). [[29, -17]] selects all systems containing copper (29), but excludes from that list any that also contain chlorine (17). [[29, 1, -17]] would select all systems that contain copper (29) and hydrogen (H), and do not include chlorine (17). Everything contain within the same brackets acts as an "and" (i.e., all criteria must be satisfied). Providing two separate sublists acts as an "or". For example, [[29,1], [78,-17]], states that a molecule can either have [copper (29) and hydrogen (1)] OR [platinum (78) and not chlorine (17)]. Leaving this field as an empty list or remove it will disable this element filtering feature.
- `regression_ase`: A boolean flag indicating whether to use the atomic self-energies provided by the dataset (if available) or to calculate them via regression. If set to `True`, the atomic self-energies will be used as provided in the dataset metadata; if set to `False`, the self-energies will be calculated via regression. This is Optional and defaults to `False`.

Other fields that can be specified in the dataset configuration file include:

- `local_yaml_file`: A path to a local dataset yaml file. This is Optional and defaults to `None`. If specified, it will be used to load the dataset metadata instead of the default metadata files provided by modelforge. This allows users to work with their own datasets without needing to upload them to a remote server or modifying the modelforge source.
- `dataset_cache_dir`: Specifies the directory where the dataset files will be cached. This is useful for storing the dataset files locally to avoid downloading them multiple times; can be shared between multiple training runs.


Processing of dataset entries
-----------------------------------

Other common operations that are performed on the dataset as part of training machine learned potentials.  These are defined in the training toml file:

- `Removing Self-Energies`: Self-energies are per-element offsets subtracted to the total energy of a system.
  The energy offsets provide cleaner training data (e.g., MAE values of energy are closer to the scale of the energy itself).
- *Splitting the Dataset*: The dataset are split into training, validation, and test sets. This is crucial for evaluating the performance of the machine learning model and ensuring that it generalizes well to unseen data. Various schemes can be used to specify this.
- *Shifting the center of mass*: The center of mass of the system can be shifted to the origin to enable calculation of the dipole moment.
- *Normalization and Scaling*: Normalize the energies and other properties to ensure they are on a comparable scale, which can improve the stability and
  performance of the machine learning model. Note that this is done when atomic energies are predicted, i.e. the atomic energy (`E_i`) is scaled using the atomic energy distribution obtained from the training dataset: `E_i = E_i_stddev * E_i_pred + E_i_mean`.

However, note that these operations are not defined within the dataset configuration; these are specified in the training (self-energy, splitting, shifting COM) and potential (normalization) configuration TOML files.

Interacting with the Dataset Module
-----------------------------------

Here, we provide a brief overview of the :class:`~modelforge.dataset.DataModule` class. Note, users will typically interact with this portion of the code indirectly via the TOML configuration files. The :class:`~modelforge.dataset.DataModule` class handles preparing and setting up datasets for training. and is designed to integrate seamlessly with PyTorch Lightning, providing a user-friendly interface for dataset preparation and loading.

The following example demonstrates how to use the :class:`~modelforge.dataset.DataModule` class to prepare and set up a dataset for training, where the similarity to the TOML configuration file should be evident.

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


yaml Metadata File Structure
------------------------------------

The :class:`~modelforge.dataset.HDF5Dataset` class is designed to provide a generic class for loading in modelforge compatible HDF5 files. This relies upon reading in a YAML file which provide essential information about a given dataset, including the available versions, properties, and other relevant details, along with the downloard url used to fetch the dataset. These YAML metadata files are stored in the `~modelforge/dataset/yaml_files` directory for the datasets provided by modelforge.

Below is a fictional example of a metadata YAML to demonstrate the key fields which includes the dataset name, version, description, atomic self-energies, and available properties.

.. code-block:: yaml

    dataset: fictional_dataset_name
    latest: full_dataset_v1.1 # an alias for the lastest version of the full dataset
    latest_test: nc_1000_v1.1 # an alias for the lastest version of the 1000 configuration test dataset

    description: "A description of the dataset."

    atomic_self_energies:
      H: -1400.0 * kilojoule_per_mole
      C: -10000.0 * kilojoule_per_mole

    full_dataset_v1.1:
      about: "This provides a curated hdf5 file for the fictional dataset designed to be compatible
        with modelforge. This dataset contains 1234 unique records for 123456 total
        configurations."
      hdf5_schema: 2 # This specifies which modelforge HDF5 schema the version uses.
      available_properties: # list of properties keys available in the dataset
      - atomic_numbers
      - positions
      - dft_energy
      remote_dataset:
        doi: 10.1234/fictional_dataset.v1.1 # The DOI for the zenodo record of the dataset
        url: https://zenodo.org/records/record_id/files/fictional_dataset_v1.1.hdf5.gz # The URL to download the gzipped HDF5 file
        gz_data_file:
          file_name: fictional_dataset_v1.1.hdf5.gz #name of the gzipped file that will be saved locally
          length: 123456 # Length of the gzipped file in bytes, used for the progress bar
          md5: gzip_checksum_value # The MD5 checksum of the gzipped file, used to verify the integrity of the downloaded file
        hdf5_data_file:
          file_name: fictional_dataset_v1.1.hdf5 # The name of the HDF5 file that will be saved locally after unzipping
          md5: hdf5_checksum_value # The MD5 checksum of the HDF5 file, used to verify the integrity of the downloaded file

Note, HDF5 datafile stored on zenodo.org are stored as gzipped files to save space and bandwidth when downloading.

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

Below is a description of the curated datasets currently available for modelforge and their corresponding metadata yaml files. These files can be found in the `~modelforge/dataset/yaml_files` directory.  The YAML files provide detailed information about each dataset, including the versions, properties, self energies and download URLs.  As previously mentioned, for each dataset, multiple versions may be available.  A 1000 configuration test dataset is provided for each dataset primarily useful for testing; several datasets also provide various subsets (e.g., limited to a subset of elements).

The dataset names used to specify the dataset in modelforge are provided in parentheses:

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