import os
from datasets import load_from_disk
from hydra.utils import instantiate


def get_loaders(config):
    train_dataset = load_from_disk(os.path.join(config.datasets.data_dir, "train"))
    train_loader = instantiate(config.dataloader, dataset=train_dataset)
    valid_dataset = load_from_disk(os.path.join(config.datasets.data_dir, "test"))
    valid_loader = instantiate(config.dataloader, dataset=valid_dataset)
    return train_loader, valid_loader
