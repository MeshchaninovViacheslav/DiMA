import os
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from typing import Optional, Dict, List
from omegaconf import DictConfig
from hydra.utils import instantiate

from src.encoders.transformer_decoder import TransformerDecoder
from src.encoders.base import Encoder


class SaprotEncoderModel(Encoder):
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
            decoder_type=main_config.decoder.decoder_type,
            add_enc_normalizer=add_enc_normalizer,
        )
        self.main_config = main_config
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.encoder_model_name)
        self.encoder = AutoModel.from_pretrained(
            config.encoder_model_name, 
            add_cross_attention=False, 
            is_decoder=False
        ).to(device)
        
        if self.decoder_type == "transformer":
            decoder_path = self.main_config.decoder.decoder_path if self.main_config else None
            self.sequence_decoder = TransformerDecoder(
                config=self.main_config
            )
            if decoder_path is not None and os.path.exists(decoder_path):
                self.sequence_decoder.load_state_dict(torch.load(decoder_path)["decoder"])
            else:
                print("Decoder wasn't initialized")
        else:
            raise NotImplementedError("Non-transformer decoder not implemented for SAProt")
        
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
        tokenized_batch = tokenized_batch.to(self.device)

        encodings = self.encoder(
            input_ids=tokenized_batch["input_ids"], 
            attention_mask=tokenized_batch["attention_mask"]
        ).last_hidden_state
        
        if self.enc_normalizer is not None:
            encodings = self.enc_normalizer.normalize(encodings)

        return encodings, tokenized_batch["attention_mask"], tokenized_batch["input_ids"]

    def batch_decode(self, encodings, attention_mask=None):
        if self.enc_normalizer is not None:
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
            logits = self.sequence_decoder(encodings)
        return logits

    def restore_decoder(self, decoder_path: str):
        if os.path.exists(decoder_path):
            self.sequence_decoder.load_state_dict(torch.load(decoder_path)["decoder"])
        else:
            print(f"Warning: Decoder checkpoint path provided, but no decoder is present in the model.")

    def pred_tokens(self, embeddings):
        output_logits = self.sequence_decoder(x=embeddings)
        predicted_token_ids = output_logits.argmax(axis=-1)
        return predicted_token_ids, output_logits
    
    def get_encodings(self, input_ids, attention_mask):
        encodings = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return encodings