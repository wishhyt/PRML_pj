"""
Loss functions for GRPO training.

This module contains different loss computation strategies as described in the paper:
- GRPO base loss (recommended): Simple policy gradient with std normalization
- GRPO with clip + KL (experimental): PPO-style clipped objective with KL penalty
- Token-level loss (experimental): DAPO-style token-level policy gradient

Paper finding: Simple GRPO loss is efficient and sufficient.
"""

import torch


def grpo_loss(
    policy_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """
    Compute basic GRPO loss (recommended by paper).
    
    This is the simplest and most effective loss function:
    L = -E[r(θ) * A], where r(θ) = π_θ / π_old
    
    Args:
        policy_logprobs: Log probs from current policy, shape (batch, seq_len)
        old_logprobs: Log probs from old policy, shape (batch, seq_len)
        advantages: Per-sample advantages, shape (batch,) or (batch, 1)
        response_mask: Mask for response tokens, shape (batch, seq_len)
        
    Returns:
        Tuple of (loss, metrics_dict)
    """
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(-1)
    
    # Compute importance sampling ratio
    ratio = torch.exp(policy_logprobs - old_logprobs)
    
    # Per-token loss
    per_token_loss = -ratio * advantages
    
    # Apply mask (only train on generated tokens)
    masked_loss = per_token_loss * response_mask
    
    # Average over sequence, then over batch
    denom = response_mask.sum(dim=-1).clamp_min(1)
    loss_per_sample = masked_loss.sum(dim=-1) / denom
    loss = loss_per_sample.mean()
    
    # Compute metrics
    with torch.no_grad():
        # Approximate KL divergence: KL(old || new) ≈ (ratio - 1) - log(ratio)
        approx_kl = ((ratio - 1) - torch.log(ratio)) * response_mask
        approx_kl_mean = approx_kl.sum() / response_mask.sum()
        
        policy_ratio_mean = (ratio * response_mask).sum() / response_mask.sum()
    
    metrics = {
        "approx_kl": approx_kl_mean.item(),
        "policy_ratio": policy_ratio_mean.item(),
    }
    
    return loss, metrics


def grpo_loss_with_clip_kl(
    policy_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_eps: float = 0.2,
    beta: float = 0.04,
) -> tuple[torch.Tensor, dict]:
    """
    Compute GRPO loss with PPO clipping and KL penalty.
    
    WARNING: Paper finding suggests this degrades performance.
    
    L = -min(r*A, clip(r, 1-ε, 1+ε)*A) + β*KL
    
    Args:
        policy_logprobs: Log probs from current policy
        old_logprobs: Log probs from old policy
        advantages: Per-sample advantages
        response_mask: Mask for response tokens
        clip_eps: PPO clip epsilon
        beta: KL penalty coefficient
        
    Returns:
        Tuple of (loss, metrics_dict)
    """
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(-1)
    
    # Compute importance sampling ratio
    ratio = torch.exp(policy_logprobs - old_logprobs)
    
    # PPO Clipped Surrogate Objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2)
    
    # KL Divergence Penalty
    approx_kl_loss = (ratio - 1) - torch.log(ratio)
    
    # Combined per-token loss
    per_token_loss = policy_loss + beta * approx_kl_loss
    
    # Apply mask and average
    masked_loss = per_token_loss * response_mask
    denom = response_mask.sum(dim=-1).clamp_min(1)
    loss_per_sample = masked_loss.sum(dim=-1) / denom
    loss = loss_per_sample.mean()
    
    # Compute metrics
    with torch.no_grad():
        approx_kl = approx_kl_loss * response_mask
        approx_kl_mean = approx_kl.sum() / response_mask.sum()
        policy_ratio_mean = (ratio * response_mask).sum() / response_mask.sum()
    
    metrics = {
        "approx_kl": approx_kl_mean.item(),
        "policy_ratio": policy_ratio_mean.item(),
    }
    
    return loss, metrics


def token_level_loss(
    policy_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    group_output_lens: torch.Tensor = None,
    group_size: int = 8,
) -> tuple[torch.Tensor, dict]:
    """
    Compute DAPO-style token-level policy gradient loss.
    
    WARNING: Paper finding suggests this does not improve performance.
    
    In sample-level loss, tokens in longer responses get smaller weights.
    Token-level loss gives longer sequences more influence on gradients.
    
    Args:
        policy_logprobs: Log probs from current policy
        old_logprobs: Log probs from old policy
        advantages: Per-sample advantages
        response_mask: Mask for response tokens
        group_output_lens: Total output length per group for normalization
        group_size: Number of samples per prompt group
        
    Returns:
        Tuple of (loss, metrics_dict)
    """
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(-1)
    
    # Compute importance sampling ratio
    ratio = torch.exp(policy_logprobs - old_logprobs)
    
    # Per-token loss
    per_token_loss = -ratio * advantages
    
    # Apply mask
    masked_loss = per_token_loss * response_mask
    
    # Token-level normalization: divide by total tokens across group
    if group_output_lens is not None:
        denom = group_output_lens.unsqueeze(-1)
        loss_per_sample = masked_loss.sum(dim=-1) / denom * group_size
    else:
        denom = response_mask.sum(dim=-1).clamp_min(1)
        loss_per_sample = masked_loss.sum(dim=-1) / denom
    
    loss = loss_per_sample.mean()
    
    # Compute metrics
    with torch.no_grad():
        approx_kl = ((ratio - 1) - torch.log(ratio)) * response_mask
        approx_kl_mean = approx_kl.sum() / response_mask.sum()
        policy_ratio_mean = (ratio * response_mask).sum() / response_mask.sum()
    
    metrics = {
        "approx_kl": approx_kl_mean.item(),
        "policy_ratio": policy_ratio_mean.item(),
    }
    
    return loss, metrics
