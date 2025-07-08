import argparse
from datasets import load_from_disk
import numpy as np

from src.utils.hydra_utils import setup_config


def main(config):
    dataset = load_from_disk(config.datasets.data_dir)["train"]["sequence"]
    lengths = [len(seq) for seq in dataset]
    quantity = np.zeros(max(lengths) + 1)
    for length in lengths:
        quantity[length] += 1
    quantity = quantity / len(lengths)
    np.save(config.datasets.length_distribution, quantity)
    print(f"Length distribution saved to {config.datasets.length_distribution}")
    print(f"Max length: {max(lengths)}")
    print(f"Min length: {min(lengths)}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config_path", type=str, default="src/configs/config.yaml")
    args = args.parse_args()
    config = setup_config(config_path=args.config_path)
    main(config)

"""
Example of usage:
python -m src.helpers.prepare_length_distribution --config_path ../../src/configs
"""