import os
import wandb
import argparse
import torch
from tqdm import tqdm
from hydra.utils import instantiate

from src.utils.hydra_utils import setup_config
from src.utils import seed_everything
from src.preprocessing.preprocessing_utils import get_loaders
from src.utils.training_utils import reconstruction_loss
from src.utils.logging_utils import config_to_wandb


def loss_step(batch, encoder, config, dynamic, eval=False):
    latent, attention_mask, input_ids = encoder.batch_encode(batch, max_sequence_len=config.datasets.max_sequence_len)
    attention_mask = attention_mask.int()

    if not eval:
        T = config.decoder.max_T
        eps = config.decoder.min_T
        t = torch.cuda.FloatTensor(latent.shape[0]).uniform_() * (T - eps) + eps
        x_t = dynamic.marginal(latent, t)["x_t"]
        latent = x_t
    
    latent = encoder.enc_normalizer.denormalize(latent)

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits = encoder.batch_get_logits(latent, attention_mask)
    loss = reconstruction_loss(input_ids, logits, mask=attention_mask)
    
    tokens = logits.argmax(dim=-1)
    acc = torch.mean((input_ids == tokens) * 1.)

    return loss, acc


def train_decoder(config, encoder, train_loader, valid_loader):
    decoder = encoder.sequence_decoder
    total_number_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Num params: {total_number_params}")

    scheduler = instantiate(config.scheduler)
    dynamic = instantiate(config.dynamic, scheduler=scheduler)

    optimizer = instantiate(config.decoder.optimizer, params=decoder.parameters())

    step = 0
    for _ in range(config.decoder.training_epochs):
        decoder.train()

        for batch in tqdm(train_loader):
            loss, acc = loss_step(
                batch=batch,
                encoder=encoder,
                config=config,
                dynamic=dynamic,
            )
       
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                decoder.parameters(),
                max_norm=1.0
            )
            optimizer.step()

            wandb.log({f'train loss': loss.item()}, step=step)
            wandb.log({f'train accuracy': acc.item()}, step=step)

            step += 1
            
        decoder.eval()
        total_loss = 0
        total_acc = 0
        total_samples = 0

        for batch in tqdm(valid_loader):
            with torch.no_grad():
                loss, acc = loss_step(
                    batch=batch,
                    encoder=encoder,
                    config=config,
                    dynamic=dynamic,
                    eval=True
                )

                total_loss += loss.item() * len(batch)
                total_acc += acc.item() * len(batch)
                total_samples += len(batch)

        wandb.log({f'valid loss': total_loss / total_samples}, step=step)
        wandb.log({f'valid accuracy': total_acc / total_samples}, step=step)

    return decoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    args = parser.parse_args()

    config = setup_config(config_path=args.config_path)

    seed_everything(config.project.seed)

    # ✅ Initialize Weights and Biases
    if not config.ddp.enabled or config.ddp.global_rank == 0:
        name = config.project.checkpoints_prefix
        wandb.init(
            project=config.project.wandb_project,
            name=name,
            mode="online"
        )
        config_to_wandb(config)

    device = torch.device("cuda:0")

    encoder_partial = instantiate(config.encoder)
    encoder = encoder_partial(
        device=device,
        main_config=config,
        add_enc_normalizer=True,
    )

    # Train decoder
    train_loader, valid_loader = get_loaders(config=config)
    decoder = train_decoder(
        config=config,
        encoder=encoder,
        train_loader=train_loader,
        valid_loader=valid_loader,
    )
    
    # Save statistics
    os.makedirs(config.project.decoder_checkpoints_folder, exist_ok=True)
    decoder.eval()

    state_dict = {
        "decoder": decoder.state_dict(),
    }
    
    path = config.decoder.decoder_path
    torch.save(state_dict, path)
    print(f"Save preprocessing to: {path}")
