import os
from datasets import load_from_disk, load_dataset
import argparse

from src.utils.hydra_utils import setup_config

# from data.load_to_hub import load_to_hub
# load_to_hub("./data", "ur50_60-254", "bayes-group-diffusion")

def load_to_hub(data_dir, dataset_name, group_name):
    dt = load_from_disk(data_dir)
    dt.push_to_hub(f"{group_name}/{dataset_name}")


def load_from_hub(data_dir, dataset_name, group_name):
    dt = load_dataset(f"{group_name}/{dataset_name}")
    dt.save_to_disk(data_dir)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config_path", type=str, default="src/configs/config.yaml")
    args.add_argument("--group_name", type=str)
    args.add_argument("--load_to_hub", action="store_true")
    args.add_argument("--load_from_hub", action="store_true")
    
    args = args.parse_args()
    config = setup_config(config_path=args.config_path)
    if args.load_to_hub:
        load_to_hub(config.datasets.data_dir, config.datasets.data_name, args.group_name)
    if args.load_from_hub:
        load_from_hub(config.datasets.data_dir, config.datasets.data_name, args.group_name)

"""
Example of usage:
python -m src.datasets.load_hub --config_path ../configs --load_to_hub --group_name="bayes-group-diffusion"
python -m src.datasets.load_hub --config_path ../configs --load_from_hub --group_name="bayes-group-diffusion"
"""