import torch
from transformers import EsmTokenizer
from typing import List
from tqdm import tqdm

from src.metrics.util import load_esm_plm


def get_model_name(model_key):
    model_classes = {
        "ESM2_15B": 'facebook/esm2_t48_15B_UR50D',   # Embedding size: 5120
        "ESM2_3B": 'facebook/esm2_t36_3B_UR50D',     # Embedding size: 2560
        "ESM2_650M": 'facebook/esm2_t33_650M_UR50D', # Embedding size: 1280
        "ESM2_150M": 'facebook/esm2_t30_150M_UR50D', # Embedding size: 640
        "ESM2_35M": 'facebook/esm2_t12_35M_UR50D',   # Embedding size: 480
        "ESM2_8M": 'facebook/esm2_t6_8M_UR50D',      # Embedding size: 320
    }

    if model_key in model_classes:
        model_name = model_classes[model_key]
        return model_name
    else:
        raise ValueError(f"Unknown model key: {model_key}")


def compute_pseudo_prob_batch(sequences, encoder, tokenizer, device, max_len):
    tokenized_X = tokenizer.batch_encode_plus(
        sequences, 
        add_special_tokens=True, 
        padding=True, 
        return_tensors='pt', 
        truncation=True,
        max_length=max_len,
    )
    input_ids = tokenized_X['input_ids'].to(device) # [batch_size, max_len]
    attention_mask = tokenized_X['attention_mask'].to(device) # [batch_size, max_len]

    batch_token_probs = torch.zeros(input_ids.shape).to(device) 

    for token_idx in range(input_ids.shape[1]):
        masked_input_ids = input_ids.clone()
        masked_input_ids[:, token_idx] = tokenizer.mask_token_id

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = encoder(input_ids=masked_input_ids, attention_mask=attention_mask)
            logits = outputs.logits # [batch_size, max_len + 2, vocab_size (33)]
            log_likelihood = torch.nn.functional.log_softmax(logits[:, token_idx, :], dim=-1) # [batch_size, vocab_size (33)]
            neg_log_likelihood = -1 * log_likelihood

        token_probabilities = torch.gather(neg_log_likelihood, -1, input_ids[:, token_idx].unsqueeze(-1)).squeeze(-1) # [batch_size]
        batch_token_probs[:, token_idx] = token_probabilities

    all_special_ids = torch.tensor(tokenizer.all_special_ids).to(device)
    expanded_input_ids = input_ids.unsqueeze(-1)
    matches = expanded_input_ids == all_special_ids
    non_special_mask = ~matches.any(dim=-1)
    sum_ll = (batch_token_probs * non_special_mask.float()).sum(dim=1) # [batch_size]
    count_tokens = non_special_mask.sum(dim=1)
    mean_ll = sum_ll / count_tokens.float()
    pppl = mean_ll.exp()

    return pppl.tolist()


def calculate_pppl(predictions: List[str], max_len: int, device: str = "cuda:0") -> List[float]:
    batch_size = 64
    model_name = get_model_name("ESM2_650M")
    tokenizer, encoder = load_esm_plm(device, model_name)

    dataset_pppl = []
    for i in tqdm(range(0, len(predictions), batch_size)):
        batch_pppl = compute_pseudo_prob_batch(predictions[i:i + batch_size], encoder, tokenizer, device, max_len)
        dataset_pppl.extend(batch_pppl)
    return dataset_pppl