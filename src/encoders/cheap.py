import torch
from omegaconf import DictConfig
from cheap.pretrained import load_pretrained_model, CHECKPOINT_DIR_PATH, get_pipeline
from cheap.proteins import LatentToSequence, DecoderTokenizer
from typing import List, Dict

from src.encoders.base import Encoder


class CHEAPEncoderModel(Encoder):
    def __init__(
            self,
            config: DictConfig,
            main_config: DictConfig = None,
            device: torch.device = None,
            add_enc_normalizer: bool = True,
    ):  
        """
        https://github.com/amyxlu/cheap-proteins/blob/7a1c43fb6834e0253b606740cfe09c164d7b18f5/src/cheap/model/_hourglass.py#L19
        """
        super().__init__(
            config=config,
            device=device,
            decoder_type=main_config.decoder.decoder_type,
            add_enc_normalizer=add_enc_normalizer,
        )
        
        self.shorten_factor = int(self.config.encoder_type.split("_")[2])
        self.channel_dimension = int(self.config.encoder_type.split("_")[4])
        
        self.tokenizer = DecoderTokenizer()
        
        self.encoder = get_pipeline(
            load_pretrained_model(
                shorten_factor=self.shorten_factor,
                channel_dimension=self.channel_dimension,
                infer_mode=True,
                model_dir=CHECKPOINT_DIR_PATH,
            )
        )
        
        self.sequence_decoder = LatentToSequence()
        self.sequence_decoder.decoder.to(self.device)
        self.sequence_decoder.device = self.device

    def batch_encode(self, batch: Dict, max_sequence_len: int):
        sequences = batch["sequence"]

        # https://github.com/amyxlu/cheap-proteins/blob/7a1c43fb6834e0253b606740cfe09c164d7b18f5/src/cheap/pipeline.py#L40
        sequences = [s[:max_sequence_len] for s in sequences]
        
        cheap_encodings, attention_mask, input_ids = self.encoder(sequences)
        
        if self.enc_normalizer is not None:
            cheap_encodings = self.enc_normalizer.normalize(cheap_encodings)
        
        return cheap_encodings, attention_mask, input_ids

    def batch_decode(self, encodings, attention_mask=None):
        if self.enc_normalizer is not None:
            encodings = self.enc_normalizer.denormalize(encodings)
        
        esm_encodings = self.encoder.decode(encodings, attention_mask)
        
        _, _, sequence = self.sequence_decoder.to_sequence(
            latent=esm_encodings, 
            mask=attention_mask,
            return_logits=True
        )
        
        sequence = [s[:int(m.sum().item() * self.shorten_factor)] for s, m in zip(sequence, attention_mask)]
        return sequence

    def get_attention_mask_for_lens(self, lens: List[int], max_sequence_len: int) -> torch.Tensor:
        max_len_in_batch = min(max(lens), max_sequence_len)
        if self.shorten_factor == 2:
            max_len_in_batch = (max_len_in_batch + 1) // 2
        
        attention_mask = torch.zeros((len(lens), max_len_in_batch), device=self.device)
        for i, l in enumerate(lens):
            if self.shorten_factor == 2:
                l = min(l, max_len_in_batch)
                l = (l + 1) // 2
            for j in range(l):
                attention_mask[i, j] = 1
        return attention_mask

    def get_decoding_artifacts(self, encodings, attention_mask):
        if self.enc_normalizer is not None:
            encodings = self.enc_normalizer.denormalize(encodings)
        
        esm_encodings = self.encoder.decode(encodings, attention_mask)
        
        _, _, sequence = self.sequence_decoder.to_sequence(
            latent=esm_encodings, 
            mask=attention_mask,
            return_logits=True
        )
        
        sequence = [s[:int(m.sum().item() * self.shorten_factor)] for s, m in zip(sequence, attention_mask)]
        return esm_encodings, sequence
    
    def get_esm_encodings(self, sequences):
        _, _, esm_encodings = self.encoder(sequences)
        return esm_encodings
    
    def batch_get_logits(self, encodings, attention_mask=None):
        if self.enc_normalizer is not None:
            encodings = self.enc_normalizer.denormalize(encodings)
        
        esm_encodings = self.encoder.decode(encodings, attention_mask)
        
        logits, _, _ = self.sequence_decoder.to_sequence(
            latent=esm_encodings, 
            mask=attention_mask,
            return_logits=True
        )
        return logits
