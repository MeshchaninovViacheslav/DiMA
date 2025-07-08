import torch
import torch.nn as nn
import math
from copy import deepcopy
from typing import List, Optional, Tuple, Union, Set, Callable

from src.models.blocks import BertBlock, timestep_embedding


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.use_self_cond = config.use_self_cond
        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.input_blocks = torch.nn.ModuleList(
            [BertBlock(config) for _ in range(0, self.num_hidden_layers // 2)]
        )
        self.output_blocks = torch.nn.ModuleList(
            [BertBlock(config) for _ in range(0, self.num_hidden_layers // 2)]
        )
        self.time_layers = torch.nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size) for _ in range(0, self.num_hidden_layers)]
        )
        if self.use_self_cond:
            self.self_cond_layers = torch.nn.ModuleList(
                [nn.Linear(self.hidden_size, self.hidden_size) for _ in range(0, self.num_hidden_layers)]
            )

    def forward(
            self,
            x: torch.Tensor,
            emb_t: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            x_0_self_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        x_input_list = []

        for i, block in enumerate(self.input_blocks):
            x_input_list.append(x)
            x = x + self.time_layers[i](emb_t)
            if self.use_self_cond and x_0_self_cond is not None:
                x = x + self.self_cond_layers[i](x_0_self_cond)
            
            x = block(
                hidden_states=x,
                attention_mask=attention_mask,
            )

        for i, block in enumerate(self.output_blocks):
            ind = i + self.num_hidden_layers // 2
            x = x + x_input_list.pop()
            x = x + self.time_layers[ind](emb_t)
            if self.use_self_cond and x_0_self_cond is not None:
                x = x + self.self_cond_layers[ind](x_0_self_cond)
            
            x = block(
                hidden_states=x,
                attention_mask=attention_mask,
            )
        return x
    

class ScoreEstimator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self._max_position_embeddings = self.config.max_position_embeddings
        
        if self.embedding_size != self.hidden_size:
            self.input_projector_x_t = torch.nn.Linear(self.embedding_size, self.hidden_size)
            self.output_projector_x_t = torch.nn.Linear(self.hidden_size, self.embedding_size)
            self.projector_self_cond = torch.nn.Linear(self.embedding_size, self.hidden_size)
        
        self.time_emb = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(self.hidden_size * 2, self.hidden_size)
        )

        self.encoder = TransformerEncoder(deepcopy(config))
        
        self.register_buffer("position_ids", torch.arange(self._max_position_embeddings).expand((1, -1)))
        self.position_embeddings = torch.nn.Embedding(self._max_position_embeddings, self.hidden_size)

    def get_extended_attention_mask(self, attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def forward(
            self,
            x_t: torch.Tensor,
            time_t: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            x_0_self_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert time_t is not None
        
        # Time embedding
        hidden_t = self.time_emb(timestep_embedding(time_t, self.hidden_size))
        hidden_t = hidden_t[:, None, :]

        # Project input if needed
        emb_x = x_t
        if self.embedding_size != self.hidden_size:
            emb_x = self.input_projector_x_t(x_t)
            if x_0_self_cond is not None:
                x_0_self_cond = self.projector_self_cond(x_0_self_cond)

        # Create positional embeddings
        seq_length = x_t.size(1)
        position_ids = self.position_ids[:, :seq_length]
        emb_pos = self.position_embeddings(position_ids)
        
        # Add positional embeddings to input
        hidden_state = emb_x + emb_pos

        if attention_mask is not None:
            attention_mask = self.get_extended_attention_mask(
                attention_mask=attention_mask,
                dtype=hidden_state.dtype
            )
        
        output = self.encoder(
            x=hidden_state,
            attention_mask=attention_mask,
            emb_t=hidden_t,
            x_0_self_cond=x_0_self_cond,
        )
        if self.embedding_size != self.hidden_size:
            output = self.output_projector_x_t(output)
        return output