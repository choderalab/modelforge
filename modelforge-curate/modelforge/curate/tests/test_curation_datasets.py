import pytest

from modelforge.curate.datasets.aimnet2_curation import Aimnet2Curation
from modelforge.curate.datasets.ani1x_curation import ANI1xCuration
from modelforge.curate.datasets.ani2x_curation import ANI2xCuration
from modelforge.curate.datasets.fe_II_curation import FeIICuration
from modelforge.curate.datasets.geom_qm9_curation import GEOMQM9Curation
from modelforge.curate.datasets.phalkethoh_curation import PhAlkEthOHCuration
from modelforge.curate.datasets.qm9_curation import QM9Curation
from modelforge.curate.datasets.spice_1_curation import SPICE1Curation
from modelforge.curate.datasets.spice_1_openff_curation import SPICE1OpenFFCuration
from modelforge.curate.datasets.spice_2_curation import SPICE2Curation
from modelforge.curate.datasets.spice_2_openff_curation import SPICE2OpenFFCuration
from modelforge.curate.datasets.tmqm_curation import tmQMCuration
from modelforge.curate.datasets.tmqm_openff_curation import tmQMOpenFFCuration
from modelforge.curate.datasets.tmqm_xtb_curation import tmQMXTBCuration

# datasets whose parameters are read from a yaml file shipped in
# modelforge/curate/datasets/yaml_files
yaml_backed_curation_classes = [
    (Aimnet2Curation, "aimnet2"),
    (ANI1xCuration, "ani1x"),
    (ANI2xCuration, "ani2x"),
    (FeIICuration, "fe_II"),
    (GEOMQM9Curation, "geom_qm9"),
    (QM9Curation, "qm9"),
    (SPICE1Curation, "spice1"),
    (SPICE2Curation, "spice2"),
    (tmQMCuration, "tmqm"),
    (tmQMXTBCuration, "tmqm_xtb"),
]

# datasets fetched from qcarchive rather than from a download url
qcarchive_curation_classes = [
    (PhAlkEthOHCuration, "PhAlkEthOH"),
    (SPICE1OpenFFCuration, "spice1_openff"),
    (SPICE2OpenFFCuration, "spice2_openff"),
    (tmQMOpenFFCuration, "tmqm_openff"),
]


@pytest.mark.parametrize(
    "curation_class, dataset_name",
    yaml_backed_curation_classes,
    ids=[dataset_name for _, dataset_name in yaml_backed_curation_classes],
)
def test_curation_class_reads_yaml_parameters(
    curation_class, dataset_name, prep_temp_dir
):
    """Instantiating a curation class must load the parameters from its yaml file."""
    local_cache_dir = str(prep_temp_dir) + f"/{dataset_name}_init"

    dataset_curation = curation_class(
        dataset_name=dataset_name, local_cache_dir=local_cache_dir
    )

    assert dataset_curation.dataset_name == dataset_name
    assert dataset_curation.version_select != "latest"
    assert isinstance(dataset_curation.dataset_download_url, str)
    assert dataset_curation.dataset_download_url.startswith("https://")
    assert isinstance(dataset_curation.dataset_md5_checksum, str)
    assert isinstance(dataset_curation.dataset_filename, str)
    assert dataset_curation.dataset_length > 0


@pytest.mark.parametrize(
    "curation_class, dataset_name",
    qcarchive_curation_classes,
    ids=[dataset_name for _, dataset_name in qcarchive_curation_classes],
)
def test_qcarchive_curation_class_initializes(
    curation_class, dataset_name, prep_temp_dir
):
    """Instantiating a qcarchive-backed curation class must set the server to query."""
    local_cache_dir = str(prep_temp_dir) + f"/{dataset_name}_init"

    dataset_curation = curation_class(
        dataset_name=dataset_name, local_cache_dir=local_cache_dir
    )

    assert dataset_curation.dataset_name == dataset_name
    assert dataset_curation.qcarchive_server.startswith("https://")
    assert dataset_curation.molecule_names == {}
