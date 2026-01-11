"""
Reward functions for GRPO training.

This module contains different reward computation strategies as described in the paper:
- Outcome reward (recommended): Simple binary correctness reward
- Length penalty (experimental): DAPO-style soft overlong punishment  
- Process reward (experimental): PRM-based process supervision

Paper finding: Only outcome reward is needed; other rewards may cause reward hacking.
"""

import torch
from typing import Optional, Callable
from math_utils import last_boxed_only_string, remove_boxed, is_equiv


def compute_outcome_reward(
    outputs: list[str],
    answers: list[str],
    group_size: int,
) -> torch.Tensor:
    """
    Compute binary outcome reward based on answer correctness.
    
    This is the recommended reward function from the paper.
    R_outcome = 1 if correct, 0 otherwise
    
    Args:
        outputs: List of generated outputs
        answers: List of ground truth answers (one per prompt, will be expanded)
        group_size: Number of outputs per prompt
        
    Returns:
        Tensor of shape (n_prompts, group_size) with 0/1 rewards
    """
    n_prompts = len(answers)
    generated_answers = [remove_boxed(last_boxed_only_string(o)) for o in outputs]
    
    rewards = [
        float(a is not None and is_equiv(a, answers[i // group_size]))
        for i, a in enumerate(generated_answers)
    ]
    
    return torch.tensor(rewards, dtype=torch.float).reshape(n_prompts, group_size)


def compute_length_penalty(
    outputs: list[str],
    tokenizer,
    expected_maximum_length: int = 1024,
    overlong_cache: int = 512,
) -> torch.Tensor:
    """
    Compute DAPO-style soft overlong punishment.
    
    WARNING: Paper finding suggests this degrades exploration.
    
    - No penalty when length <= (L_max - L_cache)
    - Linearly decreases to -1 when length goes from (L_max - L_cache) to L_max
    - Saturates at -1 when length > L_max
    
    Args:
        outputs: List of generated outputs
        tokenizer: Tokenizer for computing token lengths
        expected_maximum_length: L_max parameter
        overlong_cache: L_cache parameter
        
    Returns:
        Tensor of shape (len(outputs),) with penalty values in [-1, 0]
    """
    def _length_penalty(length_tokens: int) -> float:
        L_max = expected_maximum_length
        L_cache = overlong_cache
        if length_tokens <= L_max - L_cache:
            return 0.0
        if length_tokens <= L_max:
            return ((L_max - L_cache) - length_tokens) / float(L_cache)
        return -1.0
    
    output_lens = [len(tokenizer.encode(o)) for o in outputs]
    penalties = [_length_penalty(L) for L in output_lens]
    
    return torch.tensor(penalties, dtype=torch.float)


def compute_process_reward(
    problems: list[str],
    responses: list[str],
    prm_model,
    prm_tokenizer,
    prm_batch_size: int = 16,
    step_token: str = "\n",
    prepare_input_fn: Callable = None,
    prepare_batch_fn: Callable = None,
    derive_rewards_fn: Callable = None,
) -> torch.Tensor:
    """
    Compute process reward using a Process Reward Model (PRM).
    
    WARNING: Paper finding suggests this may lead to reward hacking.
    
    Args:
        problems: List of problem texts
        responses: List of response texts
        prm_model: Process Reward Model
        prm_tokenizer: Tokenizer for PRM
        prm_batch_size: Batch size for PRM inference
        step_token: Step delimiter for PRM
        prepare_input_fn: Function to prepare single input
        prepare_batch_fn: Function to prepare batch for model
        derive_rewards_fn: Function to extract step rewards
        
    Returns:
        Tensor of shape (len(responses),) with reward values in [0, 1]
    """
    all_scores = []
    
    for start in range(0, len(responses), prm_batch_size):
        end = min(start + prm_batch_size, len(responses))
        p_batch = problems[start:end]
        r_batch = responses[start:end]
        
        processed = [
            prepare_input_fn(p, r, tokenizer=prm_tokenizer, step_token=step_token)
            for p, r in zip(p_batch, r_batch)
        ]
        input_ids, steps, reward_flags = zip(*processed)
        
        input_ids, attention_mask, reward_flags = prepare_batch_fn(
            input_ids, reward_flags, prm_tokenizer.pad_token_id
        )
        
        # Move inputs to the same device as the PRM model
        prm_device = next(prm_model.parameters()).device
        input_ids = input_ids.to(prm_device)
        attention_mask = attention_mask.to(prm_device)
        
        with torch.inference_mode():
            _, _, rewards = prm_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_probs=True,
            )
        
        step_rewards = derive_rewards_fn(rewards, reward_flags)
        
        # Aggregate step rewards -> one scalar per response
        for sr in step_rewards:
            if sr is None or len(sr) == 0:
                all_scores.append(0.0)
            else:
                all_scores.append(float(sum(sr) / len(sr)))
    
    return torch.tensor(all_scores, dtype=torch.float)


def compute_advantages(
    rewards: torch.Tensor,
    use_std_norm: bool = True,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Compute GRPO-style group-relative advantages.
    
    Args:
        rewards: Tensor of shape (n_prompts, group_size) with rewards
        use_std_norm: Whether to normalize by std (recommended by paper)
        eps: Small constant for numerical stability
        
    Returns:
        Tensor of shape (n_prompts * group_size,) with advantages
    """
    means = rewards.mean(dim=-1, keepdim=True)
    
    if use_std_norm:
        stds = rewards.std(dim=-1, keepdim=True)
        advantages = (rewards - means) / (stds + eps)
    else:
        advantages = rewards - means
    
    return advantages.reshape(-1)
