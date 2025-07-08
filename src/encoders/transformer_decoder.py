import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
    

class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        base_config = AutoConfig.from_pretrained(config.encoder.config.encoder_model_name)
        base_config.num_hidden_layers = config.decoder.num_hidden_layers

        self._max_position_embeddings = base_config.max_position_embeddings
        self.register_buffer("position_ids", torch.arange(self._max_position_embeddings).expand((1, -1)))
        self.position_embeddings = torch.nn.Embedding(self._max_position_embeddings, base_config.hidden_size)
        
        self.transformer = AutoModel.from_config(base_config).encoder
        self.fc = nn.Linear(base_config.hidden_size, base_config.vocab_size)

    def forward(self, x, mask=None):
        if mask is not None:
            mask = self.get_extended_attention_mask(mask, x.dtype)
        seq_length = x.size(1)
        position_ids = self.position_ids[:, : seq_length]
        emb_pos = self.position_embeddings(position_ids)
        x = x + emb_pos
        return self.fc(self.transformer(x, attention_mask=mask).last_hidden_state)
    
    def get_extended_attention_mask(self, attention_mask, dtype):
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask