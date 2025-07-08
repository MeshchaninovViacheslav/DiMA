import hydra
import torch
import wandb
import torch.distributed as dist
from src.diffusion.base_trainer import BaseDiffusionTrainer
from src.utils import seed_everything, setup_ddp, print_config
from src.utils.logging_utils import config_to_wandb


@hydra.main(version_base=None, config_path="src/configs", config_name="config")
def main(config):
    # ✅ DDP (Distributed Data Parallel) setup
    if config.ddp.enabled:
        config.ddp.local_rank, config.ddp.global_rank = setup_ddp()
        config.training.batch_size_per_gpu = config.training.batch_size // dist.get_world_size()
        config.dataloader.batch_size = config.training.batch_size_per_gpu
    
    config.model.config.embedding_size = config.encoder.config.embedding_dim

    if config.ddp.global_rank == 0:
        print_config(config)

    # ✅ Seed everything
    seed = config.project.seed + config.ddp.global_rank
    seed_everything(seed)

    # ✅ Initialize Weights and Biases
    if not config.ddp.enabled or config.ddp.global_rank == 0:
        name = config.project.checkpoints_prefix
        wandb.init(
            project=config.project.wandb_project,
            name=name,
            mode="online"
        )
        config_to_wandb(config)

    device = torch.device(f"cuda:{config.ddp.local_rank}") if config.ddp.enabled else torch.device("cuda")
    trainer = BaseDiffusionTrainer(config, device)
    trainer.train()

    if config.ddp.global_rank == 0:
        wandb.finish()

if __name__ == "__main__":
    main()