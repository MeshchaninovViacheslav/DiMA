import argparse
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from src.utils.hydra_utils import setup_config


def main(config, fasta_path):
    with open(fasta_path, "r") as f:
        sequences = []
        for line in f:
            if not line.startswith(">"):
                sequences.append(line.strip())

    train_sequences, test_sequences = train_test_split(sequences, test_size=10000, random_state=42)

    dt = DatasetDict({
        "train": Dataset.from_list([{"sequence": seq} for seq in train_sequences]),
        "test": Dataset.from_list([{"sequence": seq} for seq in test_sequences])
    })
    dt.save_to_disk(config.datasets.data_dir)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config_path", type=str, default="src/configs/config.yaml")
    args.add_argument("--fasta_path", type=str)
    args = args.parse_args()
    config = setup_config(config_path=args.config_path)
    main(config, args.fasta_path)

"""
Example of usage:
python -m src.datasets.fasta_to_dt --config_path ../../src/configs --fasta_path ./data/AFDBv4_90.64-510.fasta
"""