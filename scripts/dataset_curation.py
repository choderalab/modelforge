def SPICE_2(
    hdf5_file_name: str,
    output_file_dir: str,
    local_cache_dir: str,
    force_download: bool = False,
    unit_testing_max_records=None,
):
    """
     This Fetches the SPICE 2 dataset from MOLSSI QCArchive and processes it into a curated hdf5 file.


     It provides both forces and energies calculated at the ωB97M-D3(BJ)/def2-TZVPPD level of theory,
     using Psi4.

    This includes the following collections from qcarchive. Collections included in SPICE 1.1.4 are annotated with
    along with the version used in  SPICE 1.1.4; while the underlying molecules are the same in a given collection,
    newer versions may have had some calculations redone, e.g., rerun calculations that failed or rerun with
    a patched version Psi4

       - 'SPICE Solvated Amino Acids Single Points Dataset v1.1'     * (SPICE 1.1.4 at v1.1)
       - 'SPICE Dipeptides Single Points Dataset v1.3'               * (SPICE 1.1.4 at v1.2)
       - 'SPICE DES Monomers Single Points Dataset v1.1'             * (SPICE 1.1.4 at v1.1)
       - 'SPICE DES370K Single Points Dataset v1.0'                  * (SPICE 1.1.4 at v1.0)
       - 'SPICE DES370K Single Points Dataset Supplement v1.1'       * (SPICE 1.1.4 at v1.0)
       - 'SPICE PubChem Set 1 Single Points Dataset v1.3'            * (SPICE 1.1.4 at v1.2)
       - 'SPICE PubChem Set 2 Single Points Dataset v1.3'            * (SPICE 1.1.4 at v1.2)
       - 'SPICE PubChem Set 3 Single Points Dataset v1.3'            * (SPICE 1.1.4 at v1.2)
       - 'SPICE PubChem Set 4 Single Points Dataset v1.3'            * (SPICE 1.1.4 at v1.2)
       - 'SPICE PubChem Set 5 Single Points Dataset v1.3'            * (SPICE 1.1.4 at v1.2)
       - 'SPICE PubChem Set 6 Single Points Dataset v1.3'            * (SPICE 1.1.4 at v1.2)
       - 'SPICE PubChem Set 7 Single Points Dataset v1.0'
       - 'SPICE PubChem Set 8 Single Points Dataset v1.0'
       - 'SPICE PubChem Set 9 Single Points Dataset v1.0'
       - 'SPICE PubChem Set 10 Single Points Dataset v1.0'
       - 'SPICE Ion Pairs Single Points Dataset v1.2'                * (SPICE 1.1.4 at v1.1)
       - 'SPICE PubChem Boron Silicon v1.0'
       - 'SPICE Solvated PubChem Set 1 v1.0'
       - 'SPICE Water Clusters v1.0'
       - 'SPICE Amino Acid Ligand v1.0

    Parameters
    ----------
    hdf5_file_name: str, required
        Name of the hdf5 file that will be generated.
    output_file_dir: str, required
        Directory where the hdf5 file will be saved.
    local_cache_dir: str, required
        Directory where the intermediate data will be saved; in this case it will be sqlite
        database files corresponding to each of the qcarchive collections used to generate the dataset.
    force_download: bool, optional, default=False
        If False, we will use the sqlite files that exist in the local_cache_dir (if they exist);
        note, this will check to ensure that all records on qcarchive exist in the local database,
        and will be downloaded if missing.
        If True, the entire dataset will be redownloaded.
    unit_testing_max_records: int, optional, default=None
        If set, only the first n records will be processed; this is useful for unit testing.

    Returns
    -------
    """
    from modelforge.curation.spice_2_curation import SPICE2Curation

    spice_2_data = SPICE2Curation(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        local_cache_dir=local_cache_dir,
    )
    if unit_testing_max_records is None:
        spice_2_data.process(force_download=force_download, n_threads=4)
    else:
        spice_2_data.process(
            force_download=force_download,
            unit_testing_max_records=unit_testing_max_records,
            n_threads=4,
        )


def SPICE_114_OpenFF(
    hdf5_file_name: str,
    output_file_dir: str,
    local_cache_dir: str,
    force_download: bool = False,
):
    """
    This fetches the SPICE 1.1.4 dataset from MOLSSI QCArchive using the OpenFF level of theory.

    All QM datapoints retrieved were generated using B3LYP-D3BJ/DZVP level of theory.
    This is the default theory used for force field development by the Open Force Field Initiative.
    This data appears as two separate records in QCArchive: ('spec_2'  and 'spec_6'),
    where 'spec_6' provides the dispersion corrections for energy and gradient.


    This includes the following qcarchive collections:
     - "SPICE Solvated Amino Acids Single Points Dataset v1.1",
     - "SPICE Dipeptides Single Points Dataset v1.2",
     - "SPICE DES Monomers Single Points Dataset v1.1",
     - "SPICE DES370K Single Points Dataset v1.0",
     - "SPICE PubChem Set 1 Single Points Dataset v1.2",
     - "SPICE PubChem Set 2 Single Points Dataset v1.2",
     - "SPICE PubChem Set 3 Single Points Dataset v1.2",
     - "SPICE PubChem Set 4 Single Points Dataset v1.2",
     - "SPICE PubChem Set 5 Single Points Dataset v1.2",
     - "SPICE PubChem Set 6 Single Points Dataset v1.2",

    Note, this does not include the following collections that are part of the main 1.1.4 release:

     - "SPICE Ion Pairs Single Points Dataset v1.1",
     - "SPICE DES370K Single Points Dataset Supplement v1.0",


    Parameters
    ----------
    hdf5_file_name: str, required
        Name of the hdf5 file that will be generated.
    output_file_dir: str, required
        Directory where the hdf5 file will be saved.
    local_cache_dir: str, required
        Directory where the intermediate data will be saved; in this case it will be sqlite
        database files corresponding to each of the qcarchive collections used to generate the dataset.
    force_download: bool, optional, default=False
        If False, we will use the sqlite files that exist in the local_cache_dir (if they exist);
        note, this will check to ensure that all records on qcarchive exist in the local database,
        and will be downloaded if missing.
        If True, the entire dataset will be downloaded even if it exists locally.


    """
    from modelforge.curation.spice_openff_curation import SPICEOpenFFCuration

    spice_dataset = SPICEOpenFFCuration(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        local_cache_dir=local_cache_dir,
    )
    spice_dataset.process(force_download=force_download)


def SPICE_114(
    hdf5_file_name: str,
    output_file_dir: str,
    local_cache_dir: str,
    force_download: bool = False,
):
    """
    This fetches the SPICE 1.1.4 dataset from Zenodo and saves it as curated hdf5 file.

    Small-molecule/Protein Interaction Chemical Energies (SPICE).
    The SPICE dataset contains 1.1 million conformations for a diverse set of small molecules,
    dimers, dipeptides, and solvated amino acids. It includes 15 elements, charged and
    uncharged molecules, and a wide range of covalent and non-covalent interactions.
    It provides both forces and energies calculated at the ωB97M-D3(BJ)/def2-TZVPPD level of theory,
    using Psi4 1.4.1 along with other useful quantities such as multipole moments and bond orders.


    Reference:
    Eastman, P., Behara, P.K., Dotson, D.L. et al. SPICE,
    A Dataset of Drug-like Molecules and Peptides for Training Machine Learning Potentials.
    Sci Data 10, 11 (2023). https://doi.org/10.1038/s41597-022-01882-6

    Dataset DOI:
    https://doi.org/10.5281/zenodo.8222043


    Parameters
    ----------
    hdf5_file_name: str, required
        Name of the hdf5 file that will be generated.
    output_file_dir: str, required
        Directory where the hdf5 file will be saved.
    local_cache_dir: str, required
        Directory where the intermediate data will be saved; in this case it will be the hdf5 file from Zenodo.
    force_download: bool, optional, default=False
        If False, we will use the hdf5 file that exists in the local_cache_dir (if it exists);
        If True, the hdf5 dataset will be redownloaded from Zenodo.org.

    Returns
    -------

    """
    from modelforge.curation.spice_114_curation import SPICE114Curation

    spice_114 = SPICE114Curation(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        local_cache_dir=local_cache_dir,
    )
    spice_114.process(force_download=force_download)


def QM9(
    hdf5_file_name: str,
    output_file_dir: str,
    local_cache_dir: str,
    force_download: bool = False,
    unit_testing_max_records=None,
):
    """
    This fetches and process the QM9 dataset into a curated hdf5 file.

    The QM9 dataset includes 133,885 organic molecules with up to nine heavy atoms (CONF).
    All properties were calculated at the B3LYP/6-31G(2df,p) level of quantum chemistry.

    Citation: Ramakrishnan, R., Dral, P., Rupp, M. et al.
                "Quantum chemistry structures and properties of 134 kilo molecules."
                Sci Data 1, 140022 (2014).
                https://doi.org/10.1038/sdata.2014.22

    DOI for dataset: 10.6084/m9.figshare.c.978904.v5
    Parameters
    ----------
    hdf5_file_name: str, required
        Name of the hdf5 file that will be generated.
    output_file_dir: str, required
        Directory where the hdf5 file will be saved.
    local_cache_dir: str, required
        Directory where the intermediate data will be saved; in this case it will be tarred file downloaded
        from figshare and the expanded archive that contains xyz files for each molecule in the dataset.
    force_download: bool, optional, default=False
        If False, we will use the tarred file that exists in the local_cache_dir (if it exists);
        If True, the tarred file will be downloaded from figshare.org, even if it exists locally.


    """
    from modelforge.curation.qm9_curation import QM9Curation

    qm9 = QM9Curation(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        local_cache_dir=local_cache_dir,
    )
    if unit_testing_max_records is None:
        qm9.process(force_download=force_download)
    else:
        qm9.process(
            force_download=force_download,
            unit_testing_max_records=unit_testing_max_records,
        )


def ANI1x(
    hdf5_file_name: str,
    output_file_dir: str,
    local_cache_dir: str,
    force_download: bool = False,
):
    """
    This fetches and processes the ANI1x dataset into a curated hdf5 file.

    This dataset includes ~5 million density function theory calculations
    for small organic molecules containing H, C, N, and O.
    A subset of ~500k are computed with accurate coupled cluster methods.

    References:

    ANI-1x dataset:
    Smith, J. S.; Nebgen, B.; Lubbers, N.; Isayev, O.; Roitberg, A. E.
    Less Is More: Sampling Chemical Space with Active Learning.
    J. Chem. Phys. 2018, 148 (24), 241733.
    https://doi.org/10.1063/1.5023802
    https://arxiv.org/abs/1801.09319

    ANI-1ccx dataset:
    Smith, J. S.; Nebgen, B. T.; Zubatyuk, R.; Lubbers, N.; Devereux, C.; Barros, K.; Tretiak, S.; Isayev, O.; Roitberg, A. E.
    Approaching Coupled Cluster Accuracy with a General-Purpose Neural Network Potential through Transfer Learning. N
    at. Commun. 2019, 10 (1), 2903.
    https://doi.org/10.1038/s41467-019-10827-4

    wB97x/def2-TZVPP data:
    Zubatyuk, R.; Smith, J. S.; Leszczynski, J.; Isayev, O.
    Accurate and Transferable Multitask Prediction of Chemical Properties with an Atoms-in-Molecules Neural Network.
    Sci. Adv. 2019, 5 (8), eaav6490.
    https://doi.org/10.1126/sciadv.aav6490


    Dataset DOI:
    https://doi.org/10.6084/m9.figshare.c.4712477.v1

    Parameters
    ----------
    hdf5_file_name: str, required
        Name of the hdf5 file that will be generated.
    output_file_dir: str, required
        Directory where the hdf5 file will be saved.
    local_cache_dir: str, required
        Directory where the intermediate data will be saved; in this case it will be the hdf5 file downloaded from figshare.
    force_download: bool, optional, default=False
        If False, we will use the hdf5 file that exists in the local_cache_dir (if it exists);
        If True, the hdf5 dataset will be redownloaded from figshare.org.

    """

    from modelforge.curation.ani1x_curation import ANI1xCuration

    ani1x = ANI1xCuration(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        local_cache_dir=local_cache_dir,
    )
    ani1x.process(force_download=force_download)


def ANI2x(
    hdf5_file_name: str,
    output_file_dir: str,
    local_cache_dir: str,
    force_download: bool = False,
    unit_testing_max_records=None,
):
    """
    This fetches and processes the ANI2x dataset into a curated hdf5 file.

    The ANI-2x data set includes properties for small organic molecules that contain
        H, C, N, O, S, F, and Cl.  This dataset contains 9651712 conformers for 200,000
        This will fetch data generated with the wB97X/631Gd level of theory
        used in the original ANI-2x paper, calculated using Gaussian 09

        Citation: Devereux, C, Zubatyuk, R., Smith, J. et al.
                    "Extending the applicability of the ANI deep learning molecular potential to sulfur and halogens."
                    Journal of Chemical Theory and Computation 16.7 (2020): 4192-4202.
                    https://doi.org/10.1021/acs.jctc.0c00121

        DOI for dataset: 10.5281/zenodo.10108941
    """
    from modelforge.curation.ani2x_curation import ANI2xCuration

    ani2x = ANI2xCuration(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        local_cache_dir=local_cache_dir,
    )
    if unit_testing_max_records is None:
        ani2x.process(force_download=force_download)
    else:
        ani2x.process(
            force_download=force_download,
            unit_testing_max_records=unit_testing_max_records,
        )


"""
Download the various datasets and process them into curated hdf5 files.
"""

# define the local path prefix
local_prefix = "/Users/cri/Documents/Projects-msk/datasets"

# we will save all the files to a central location
output_file_dir = f"{local_prefix}/hdf5_files"

# ANI2x test dataset
# local_cache_dir = f"{local_prefix}/ani2x_dataset"
# hdf5_file_name = "ani2x_dataset.hdf5"
#
# ANI2x(
#     hdf5_file_name,
#     output_file_dir,
#     local_cache_dir,
#     force_download=False,
#     # unit_testing_max_records=100,
# )

# # QM9 dataset
# local_cache_dir = f"{local_prefix}/qm9_dataset"
# hdf5_file_name = "qm9_dataset_n100.hdf5"
#
# QM9(
#     hdf5_file_name,
#     output_file_dir,
#     local_cache_dir,
#     force_download=False,
#     unit_testing_max_records=100,
# )

# # SPICE 2 dataset
local_cache_dir = f"{local_prefix}/spice2_dataset"
hdf5_file_name = "spice_2_dataset.hdf5"

SPICE_2(hdf5_file_name, output_file_dir, local_cache_dir, force_download=False)

# # SPICE 1.1.4 OpenFF dataset
# local_cache_dir = f"{local_prefix}/spice_openff_dataset"
# hdf5_file_name = "spice_114_openff_dataset.hdf5"
#
# SPICE_114_OpenFF(hdf5_file_name, output_file_dir, local_cache_dir, force_download=False)
#
# # SPICE 1.1.4 dataset
# local_cache_dir = f"{local_prefix}/spice_114_dataset"
# hdf5_file_name = "spice_114_dataset.hdf5"
#
# SPICE_114(hdf5_file_name, output_file_dir, local_cache_dir, force_download=False)
# #
# # QM9 dataset
# local_cache_dir = f"{local_prefix}/qm9_dataset"
# hdf5_file_name = "qm9_dataset.hdf5"
#
# QM9(hdf5_file_name, output_file_dir, local_cache_dir, force_download=False)
#
# # ANI-1x dataset
# local_cache_dir = f"{local_prefix}/ani1x_dataset"
# hdf5_file_name = "ani1x_dataset.hdf5"
#
# ANI1x(hdf5_file_name, output_file_dir, local_cache_dir, force_download=True)
#
# # ANI-2x dataset
# local_cache_dir = f"{local_prefix}/ani2x_dataset"
# hdf5_file_name = "ani2x_dataset.hdf5"
#
# ANI2x(hdf5_file_name, output_file_dir, local_cache_dir, force_download=False)
