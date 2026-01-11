"""
Tokenization utilities for GRPO training.
"""

import torch
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """
    Tokenize prompts and outputs for training.
    
    Args:
        prompt_strs: List of prompt strings
        output_strs: List of output/response strings
        tokenizer: Tokenizer to use
        
    Returns:
        Dictionary with:
            - input_ids: shape (batch, max_len - 1)
            - labels: shape (batch, max_len - 1), shifted by 1
            - response_mask: bool tensor, 1 for generated tokens
    """
    prompt_t = [tokenizer.encode(p) for p in prompt_strs]
    output_t = [tokenizer.encode(o) for o in output_strs]
    
    # Find max length
    max_len = 0
    for i in range(len(prompt_t)):
        row_len = len(prompt_t[i]) + len(output_t[i])
        max_len = max(max_len, row_len)
    
    # Pad and create tensors
    full = []
    for i in range(len(prompt_t)):
        padding_size = max_len - len(prompt_t[i]) - len(output_t[i])
        padding = [tokenizer.pad_token_id] * padding_size
        row = torch.tensor(prompt_t[i] + output_t[i] + padding, dtype=torch.long)
        full.append(row.unsqueeze(0))
    
    f2 = torch.cat(full)
    input_ids = f2[:, :-1]
    labels = f2[:, 1:]
    
    # Create response mask
    response_mask = torch.zeros(len(prompt_strs), max_len - 1)
    for i in range(len(prompt_t)):
        response_mask[
            i, len(prompt_t[i]) - 1 : len(prompt_t[i]) + len(output_t[i]) - 1
        ] = 1
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask.bool(),
    }


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Compute log probabilities for each token in the response.
    
    Args:
        model: Language model
        input_ids: Input token IDs, shape (batch, seq_len)
        labels: Label token IDs (shifted), shape (batch, seq_len)
        
    Returns:
        Log probabilities for each label token, shape (batch, seq_len)
    """
    logits = model(input_ids).logits
    
    # Numerically stable log softmax
    z = logits - logits.max(dim=-1, keepdim=True).values
    exp_z = torch.exp(z)
    denom = exp_z.sum(dim=-1, keepdim=True)
    logprobs = z - torch.log(denom)
    
    # Gather log probs for the actual labels
    logprobs_for_label = torch.gather(
        logprobs, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)
    
    return logprobs_for_label
