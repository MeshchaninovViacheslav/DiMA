import wandb
import torch
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List, Union
import os


def print_config(cfg: DictConfig):
    """Print Hydra configuration accurately."""
    
    print(OmegaConf.to_yaml(cfg, resolve=False)) 


def config_to_wandb(cfg: DictConfig):
    os.makedirs("wandb_artifacts", exist_ok=True)
    config_path = "wandb_artifacts/config.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Log the config file as an artifact in W&B
    artifact = wandb.Artifact(name="hydra_config", type="config")
    artifact.add_file(config_path)
    wandb.log_artifact(artifact)


def log_batch_of_tensors_to_wandb(batch_of_tensors: Dict[str, torch.Tensor]):
    """Log a batch of tensors to W&B."""
    batch_index = 0
    columns = sorted(batch_of_tensors.keys())
    seq_len = batch_of_tensors[columns[0]].shape[1]
    data = [tuple(batch_of_tensors[col][batch_index][i].detach().cpu().item() for col in columns) for i in range(seq_len)]

    table = wandb.Table(columns=columns, data=data)
    wandb.log({"token_table": table})


def log_batch_of_texts_to_wandb(batch_of_texts: List[str]):
    # Convert list of strings to list of tuples for wandb.Table
    table = wandb.Table(columns=["text"])
    for text in batch_of_texts:
        table.add_data(text)
    wandb.log({"generated_texts": table})

def log_metric(metric_name: str, loader_name: str, value: Union[float, torch.Tensor, wandb.Image], step: int):    
    wandb.log({f'{metric_name}/{loader_name}': value}, step=step)