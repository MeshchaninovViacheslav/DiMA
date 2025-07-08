# In a new file, e.g., src/models/dima_model.py

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from pathlib import Path
import requests
from tqdm import tqdm
import os
import json
from omegaconf import DictConfig

from src.diffusion.base_trainer import BaseDiffusionTrainer
from src.utils.hydra_utils import setup_config
from src.utils.pretrained_utils import PRETRAINED_MODELS_PATHS

# The base URL for your "model hub" index file
MODEL_INDEX_URL = "https://your-cloud-service.com/models/model_index.json"

class DiMAModel(BaseDiffusionTrainer):
    def __init__(self, config_path: str, device: torch.device):
        config = setup_config(config_path=config_path)
        super().__init__(config, device)
        
    def _get_file_or_download(self, relative_path: str) -> Path:
        """
        Checks for a local file at a path relative to the project root. 
        If it doesn't exist, downloads it from the configured S3 bucket.

        Args:
            relative_path (str): The file path relative to the project root.

        Returns:
            Path: The validated local path to the file.
            
        Raises:
            ValueError: If S3 is not configured in the config file.
            FileNotFoundError: If the file is not found locally or in S3.
            IOError: If any other download error occurs.
        """
        # Project path is sourced from the main config file
        local_path = Path(self.config.project.path) / relative_path
        
        if local_path.exists():
            print(f"Found local file: {local_path}")
            return local_path
            
        print(f"Local file not found. Attempting to download from S3: {relative_path}")
        
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        s3_config = self.config.get('s3')
        if not s3_config or not s3_config.get('bucket'):
            raise ValueError("S3 bucket information must be configured in your config file.")
            
        bucket_name = s3_config.bucket
        region = s3_config.get('region')
        s3_key = relative_path

        if region and region != "us-east-1":
            url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{s3_key}"
        else:
            url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"

        try:
            self._download_file(url, local_path)
            return local_path
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise FileNotFoundError(
                    f"Could not find file locally ('{local_path}') or in S3 ('{url}'). "
                    "Please check the path and ensure the file is uploaded."
                ) from e
            else:
                raise IOError(f"Failed to download file from {url}. Status code: {e.response.status_code}") from e

    def load_model_weights(
        self, 
        diffusion_ckpt_path: str, 
        decoder_ckpt_path: str = None, 
        stats_path: str = None
    ):
        """
        Loads all necessary model weights and statistics.

        For each path provided, the method first checks for a local file. 
        If not found, it attempts to download the file from the S3 bucket specified
        in the project's configuration. Paths should be relative to the project root.

        Args:
            diffusion_ckpt_path (str): Relative path to the diffusion model checkpoint.
            decoder_ckpt_path (str, optional): Relative path to the decoder model checkpoint. Defaults to None.
            stats_path (str, optional): Relative path to the statistics file. Defaults to None.
        """
        # Load Diffusion model checkpoint
        diffusion_local_path = self._get_file_or_download(diffusion_ckpt_path)
        self.restore_checkpoint(diffusion_local_path)
        

        # Load Decoder model checkpoint if path is provided
        if decoder_ckpt_path:
            decoder_local_path = self._get_file_or_download(decoder_ckpt_path)
            print(f"Loading decoder model from {decoder_local_path}")
            self.encoder.restore_decoder(decoder_local_path)
        
        # Load statistics if path is provided
        if stats_path:
            stats_local_path = self._get_file_or_download(stats_path)
            print(f"Loading statistics from {stats_local_path}")
            self.encoder.enc_normalizer._load_state_dict(stats_local_path)
    
    def load_pretrained(self):
        encoder_name = self.config.encoder.config.encoder_type
        if encoder_name not in PRETRAINED_MODELS_PATHS:
            raise ValueError(f"Encoder {encoder_name} not supported. Supported encoders: {PRETRAINED_MODELS_PATHS.keys()}")

        diffusion_local_path = self._get_file_or_download(PRETRAINED_MODELS_PATHS[encoder_name]["diffusion"])
        self.restore_checkpoint(diffusion_local_path)
        stats_local_path = self._get_file_or_download(PRETRAINED_MODELS_PATHS[encoder_name]["stats"])
        self.encoder.enc_normalizer._load_state_dict(stats_local_path)

        if encoder_name != "CHEAP_shorten_1_dim_1024":
            decoder_local_path = self._get_file_or_download(PRETRAINED_MODELS_PATHS[encoder_name]["decoder"])
            self.encoder.restore_decoder(decoder_local_path)

    @staticmethod
    def _download_file(url: str, destination: Path):
        """Downloads a file with a progress bar, checking if it already exists."""
        if destination.exists():
            print(f"File already cached: {destination}")
            return

        print(f"Downloading {url} to {destination}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f, tqdm(
            desc=destination.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
