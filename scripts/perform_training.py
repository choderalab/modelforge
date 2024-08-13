# This script provides a command line interface to train an neurtal network potential.

import argparse
from modelforge.train.training import read_config_and_train
from typing import Union, List
from modelforge.utils.io import parse_devices

parse = argparse.ArgumentParser(description="Perform Training Using Modelforge")

parse.add_argument(
    "--condensed_config_path", type=str, help="Path to the condensed TOML config file"
)
parse.add_argument(
    "--training_parameter_path", type=str, help="Path to the training TOML config file"
)
parse.add_argument(
    "--dataset_parameter_path", type=str, help="Path to the dataset TOML config file"
)
parse.add_argument(
    "--potential_parameter_path",
    type=str,
    help="Path to the potential TOML config file",
)
parse.add_argument(
    "--runtime_parameter_path", type=str, help="Path to the runtime TOML config file"
)
parse.add_argument("--accelerator", type=str, help="Accelerator to use for training")
parse.add_argument(
    "--devices", type=parse_devices, help="Device(s) to use for training"
)
parse.add_argument(
    "--number_of_nodes", type=int, help="Number of nodes to use for training"
)
parse.add_argument("--experiment_name", type=str, help="Name of the experiment")
parse.add_argument("--save_dir", type=str, help="Directory to save the model")
parse.add_argument("--local_cache_dir", type=str, help="Local cache directory")
parse.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint")
parse.add_argument("--log_every_n_steps", type=int, help="Log every n steps")
parse.add_argument(
    "--simulation_environment",
    type=str,
    help="Simulation environment to use for training",
)


args = parse.parse_args()
read_config_and_train(
    condensed_config_path=args.condensed_config_path,
    training_parameter_path=args.training_parameter_path,
    dataset_parameter_path=args.dataset_parameter_path,
    potential_parameter_path=args.potential_parameter_path,
    runtime_parameter_path=args.runtime_parameter_path,
    accelerator=args.accelerator,
    devices=args.devices,
    number_of_nodes=args.number_of_nodes,
    experiment_name=args.experiment_name,
    save_dir=args.save_dir,
    local_cache_dir=args.local_cache_dir,
    checkpoint_path=args.checkpoint_path,
    log_every_n_steps=args.log_every_n_steps,
    simulation_environment=args.simulation_environment,
)
