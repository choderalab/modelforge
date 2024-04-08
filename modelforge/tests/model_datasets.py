from modelforge.curation.model_dataset import ModelDataset

dataset = ModelDataset(
    hdf5_file_name="PURE_MM.hdf5",
    output_file_dir="/Users/cri/Dropbox/data_experiment/",
    local_cache_dir="/Users/cri/Dropbox/data_experiment/",
    convert_units=True,
)
dataset.process(
    input_data_path="/Users/cri/Dropbox/data_experiment/",
    input_data_file="molecule_data.hdf5",
    data_combination="PURE_MM",
)

dataset = ModelDataset(
    hdf5_file_name="PURE_ML.hdf5",
    output_file_dir="/Users/cri/Dropbox/data_experiment/",
    local_cache_dir="/Users/cri/Dropbox/data_experiment/",
    convert_units=True,
)
dataset.process(
    input_data_path="/Users/cri/Dropbox/data_experiment/",
    input_data_file="molecule_data.hdf5",
    data_combination="PURE_ML",
)


dataset = ModelDataset(
    hdf5_file_name="MM_low_e_correction.hdf5",
    output_file_dir="/Users/cri/Dropbox/data_experiment/",
    local_cache_dir="/Users/cri/Dropbox/data_experiment/",
    convert_units=True,
)
dataset.process(
    input_data_path="/Users/cri/Dropbox/data_experiment/",
    input_data_file="molecule_data.hdf5",
    data_combination="PURE_MM_low_temp_correction",
)
