import os
import torch
from typing import Optional, Dict, List
from omegaconf import DictConfig

from esm.models.esmc import ESMC

from .enc_normalizer import EncNormalizer
from .decoder import Decoder
from .base import Encoder


class ESMCEncoderModel(Encoder):
    def __init__(
            self,
            config: DictConfig,
            main_config: DictConfig = None,
            device: torch.device = None,
            add_enc_normalizer: bool = True,
    ):  
        super().__init__(
            config=config,
            device=device,
            decoder_type=main_config.decoder.decoder_type if main_config else None,
            add_enc_normalizer=add_enc_normalizer,
        )
        self.main_config = main_config
        
        self.encoder = ESMC.from_pretrained(config.encoder_model_name)
        self.tokenizer = self.encoder.tokenizer
        
        if self.decoder_type == "transformer":
            decoder_path = self.main_config.decoder.decoder_path
            self.sequence_decoder = Decoder(config=self.main_config, vocab_size=self.tokenizer.vocab_size)
            if decoder_path is not None and os.path.exists(decoder_path):
                self.sequence_decoder.load_state_dict(torch.load(decoder_path)["decoder"])
            else:
                print("Decoder wasn't initialized")
        else:
            self.sequence_decoder = None
            
        if device is not None:
            self.encoder = self.encoder.to(device)
            if self.sequence_decoder is not None:
                self.sequence_decoder = self.sequence_decoder.to(device)

    def batch_encode(self, batch: Dict, max_sequence_len: int):
        max_len_with_special_tokens = max_sequence_len + 2

        sequences = batch["sequence"]
        tokenized_batch = self.tokenizer(
            sequences, 
            return_attention_mask=True, 
            return_tensors="pt", 
            truncation=True,                       
            padding=True, 
            max_length=max_len_with_special_tokens,
            return_special_tokens_mask=True,
        )
        tokenized_batch = {k: v.to(self.device) for k, v in tokenized_batch.items()}

        encodings = self.encoder(
            sequence_tokens=tokenized_batch["input_ids"],
            sequence_id=tokenized_batch["attention_mask"]
        ).embeddings
        
        if self.enc_normalizer is not None:
            encodings = self.enc_normalizer.normalize(encodings)

        return encodings, tokenized_batch["attention_mask"], tokenized_batch["input_ids"]

    def batch_decode(self, encodings, attention_mask=None):
        encodings = self.enc_normalizer.denormalize(encodings)
        logits = self.sequence_decoder(x=encodings, mask=attention_mask)

        token_ids = logits.argmax(axis=-1).detach().cpu().tolist()
        if attention_mask is not None:
            for i, t in enumerate(token_ids):
                seq_len = int(attention_mask[i].sum().item())
                token_ids[i] = t[:seq_len]

        token_ids = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        decoded_sequences = [''.join(t.split()) for t in token_ids]
        return decoded_sequences

    def batch_get_logits(self, encodings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.enc_normalizer is not None:
            encodings = self.enc_normalizer.denormalize(encodings)
        if self.decoder_type == "transformer":
            logits = self.sequence_decoder(x=encodings, mask=attention_mask)
        else:
            raise NotImplementedError("Non-transformer decoder is not implemented for ESMC")
        return logits

    def restore_decoder(self, decoder_path: str):
        if os.path.exists(decoder_path):
            self.sequence_decoder.load_state_dict(torch.load(decoder_path)["decoder"])
        else:
            print(f"Warning: Decoder checkpoint path provided, but no decoder is present in the model.")