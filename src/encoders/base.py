import os
import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import List, Optional, Tuple, Dict

from src.encoders.enc_normalizer import EncNormalizer


class Encoder(nn.Module):
    """
    Base class for all encoders.
    The desired interface consists of two main methods:
    
    - batch_encode(sequences: List[str], max_sequence_len: int) -> Tuple[torch.Tensor, torch.Tensor]
        Encodes a batch of sequences into latent representations and attention masks
        
    - batch_decode(encodings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> List[str]
        Decodes latent representations back into sequences, optionally using attention masks
    """
    def __init__(self, 
                 config: DictConfig, 
                 device: torch.device,
                 decoder_type: str,
                 add_enc_normalizer: bool = True,
            ):
        super().__init__()
        self.config = config
        self.device = device

        if add_enc_normalizer:
            self.enc_normalizer = EncNormalizer(
                enc_path=self.config.statistics_path,
            )
        else:
            self.enc_normalizer = None
        self.decoder_type = decoder_type

    def batch_encode(self, batch: Dict, max_sequence_len: int) -> torch.Tensor:
        """
        max_sequence_len: int
            The maximum sequence length of a protein. So if your encoder needs special tokens, 
            don't forget to include them in the max_sequence_len separately.
        """
        pass

    def batch_decode(self, encodings: torch.Tensor) -> List[str]:
        pass

    def get_attention_mask_for_lens(self, lens: List[int], max_sequence_len: int) -> torch.Tensor:
        max_len_with_special_tokens = max_sequence_len + 2
        max_len_in_batch = min(max(lens), max_len_with_special_tokens)
        
        attention_mask = torch.zeros((len(lens), max_len_in_batch), device=self.device)
        for i, l in enumerate(lens):
            for j in range(min(l, max_len_in_batch)):
                attention_mask[i, j] = 1
        return attention_mask

    def batch_get_logits(self, encodings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass

    def restore_decoder(self, decoder_path: str):
        pass