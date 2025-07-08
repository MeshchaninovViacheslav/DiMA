import torch
from transformers import T5Tokenizer, T5EncoderModel, EsmTokenizer, EsmForMaskedLM
import numpy as np
from typing import List
import re
from tqdm import tqdm


def load_t5_plm(device):
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", torch_dtype=torch.float16)
    encoder = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', torch_dtype=torch.float16).to(device)
    encoder.eval()
    return tokenizer, encoder


def load_esm_plm(device, model_name):
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    encoder = EsmForMaskedLM.from_pretrained(model_name, add_cross_attention=False, is_decoder=False).to(device)
    encoder.eval()
    return tokenizer, encoder


def create_t5_embeds(encoder, tokenizer, raw_seq_list, device, max_len, batch_size=256):
    seq_list = []
    len_seqs = []
    for seq in raw_seq_list:
        init_seq = " ".join(list(re.sub(r"[UZOB]", "X", seq)))
        seq_list.append(init_seq)
        len_seqs.append(len(seq))
    if max_len:
        inputs = tokenizer(seq_list, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
    else:
        inputs = tokenizer(seq_list, return_tensors="pt", padding=True).to(device)

    embeddings = np.zeros((len(seq_list), 1024))
    for i in range(0, inputs.input_ids.shape[0], batch_size):
        batch = {key: inputs[key][i:i+batch_size, :].to(device) for key in inputs.keys()}
        
        with torch.no_grad(), torch.autocast("cuda"):
            batch_embeddings = encoder(**batch).last_hidden_state
        mask = batch["attention_mask"]
        batch_embeddings = torch.sum(batch_embeddings * mask[..., None], dim=1) / torch.sum(mask, dim=1)[..., None]
        batch_embeddings = batch_embeddings.detach().cpu().numpy()

        embeddings[i:i+batch_size, :] = batch_embeddings
    return embeddings


def create_embeds(seq_list_1, seq_list_2, max_len, device="cuda:0"):
    tokenizer, encoder = load_t5_plm(device)
    embeddings_1 = create_t5_embeds(
        encoder, 
        tokenizer, 
        seq_list_1, 
        device, 
        max_len=max_len,
    )
    embeddings_2 = create_t5_embeds(
        encoder, 
        tokenizer, 
        seq_list_2, 
        device, 
        max_len=max_len,
    )
    return embeddings_1, embeddings_2