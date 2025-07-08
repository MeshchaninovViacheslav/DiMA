import os
import torch
from tqdm import tqdm
import argparse
from hydra.utils import instantiate

from src.utils.hydra_utils import setup_config
from src.utils import seed_everything
from src.preprocessing.preprocessing_utils import get_loaders

def compute_mean_std(
        config,
        encoder,
        train_loader,
        device,
):
    encoder.eval()

    encodings_sum = torch.zeros(config.encoder.config.embedding_dim, device=device)
    encodings_sum_of_squares = torch.zeros(config.encoder.config.embedding_dim, device=device)
    encodings_count = torch.Tensor([0.0]).to(device)
    
    T = tqdm(train_loader)

    for i, batch in enumerate(T):
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            clean_X, attention_mask, _ = encoder.batch_encode(batch, max_sequence_len=config.datasets.max_sequence_len)
            attention_mask = attention_mask.int()

        clean_X = clean_X.reshape(-1, clean_X.shape[-1])
        mask = attention_mask.reshape(-1).bool()
        clean_X = clean_X[mask]

        encodings_sum += torch.sum(clean_X, dim=0)
        encodings_sum_of_squares += torch.sum(clean_X ** 2, dim=0)
        encodings_count += torch.sum(mask)

        if i == 100:
            break

    encodings_mean = encodings_sum / encodings_count
    encodings_sqr = (encodings_sum_of_squares / encodings_count - encodings_mean ** 2)
    encodings_std = torch.sqrt(torch.clip(encodings_sqr, min=1e-4))
    return encodings_mean, encodings_std


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    args = parser.parse_args()

    config = setup_config(config_path=args.config_path)

    seed_everything(config.project.seed)

    device = torch.device("cuda:0")

    encoder_partial = instantiate(config.encoder)
    encoder = encoder_partial(
        device=device,
        main_config=config,
        add_enc_normalizer=False,
    )

    # Calculate statistics
    train_loader, _ = get_loaders(config=config)

    enc_mean, enc_std = compute_mean_std(
        config=config,
        encoder=encoder,
        train_loader=train_loader,
        device=device,
    )

    os.makedirs(config.project.statistics_folder, exist_ok=True)

    state_dict = {
        "enc_mean": enc_mean,
        "enc_std": enc_std,
    }

    path = config.encoder.config.statistics_path
    torch.save(state_dict, path)
    print(f"Save statistics to: {path}")