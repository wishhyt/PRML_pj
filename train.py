#!/usr/bin/env python3
"""
Unified training script for GRPO fine-tuning.

This script implements the experiments from the paper:
"Occam's Razor for RL Fine-Tuning: The Simple Design Is Efficient and Sufficient"

Usage:
    # Run with paper's recommended (best) configuration
    python train.py
    
    # Experiment 1: LoRA rank comparison
    python train.py --lora_r 1
    python train.py --lora_r 4
    python train.py --lora_r 16
    
    # Experiment 2: RL techniques
    python train.py --use_kl_penalty --beta 0.04
    python train.py --use_clip --clip_eps 0.2
    python train.py --use_token_level_loss
    
    # Experiment 3: Reward design
    python train.py --use_length_penalty
    python train.py --use_prm --process_reward_weight 0.5
"""

import argparse
from src.trainer import GRPOTrainer, TrainerConfig


def parse_args() -> TrainerConfig:
    parser = argparse.ArgumentParser(
        description="GRPO training for LLM fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda:0")
    
    # LoRA
    parser.add_argument("--lora_r", type=int, default=1, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    # Training
    parser.add_argument("--lr", type=float, default=9e-5)
    parser.add_argument("--n_grpo_steps", type=int, default=50)
    parser.add_argument("--n_prompts_per_step", type=int, default=32)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--epochs_per_step", type=int, default=1)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    
    # GRPO techniques (paper: keep these OFF)
    parser.add_argument("--use_std_norm", action="store_true", default=True,
                        help="Use std normalization (recommended ON)")
    parser.add_argument("--no_std_norm", dest="use_std_norm", action="store_false")
    parser.add_argument("--use_kl_penalty", action="store_true", default=False,
                        help="Use KL penalty (paper: degrades performance)")
    parser.add_argument("--use_clip", action="store_true", default=False,
                        help="Use PPO clipping (paper: degrades performance)")
    parser.add_argument("--beta", type=float, default=0.04, help="KL coefficient")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="Clip epsilon")
    
    # Reward design (paper: keep these OFF)
    parser.add_argument("--use_length_penalty", action="store_true", default=False,
                        help="Use length penalty (paper: degrades exploration)")
    parser.add_argument("--use_prm", action="store_true", default=False,
                        help="Use process reward model (paper: causes reward hacking)")
    parser.add_argument("--expected_maximum_length", type=int, default=1024)
    parser.add_argument("--overlong_cache", type=int, default=512)
    parser.add_argument("--process_reward_weight", type=float, default=0.5)
    
    # Loss
    parser.add_argument("--use_token_level_loss", action="store_true", default=False,
                        help="Use token-level loss (paper: no improvement)")
    
    # PRM
    parser.add_argument("--prm_model_id", type=str, 
                        default="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B")
    parser.add_argument("--prm_device", type=str, default="auto")
    parser.add_argument("--prm_batch_size", type=int, default=16)
    
    # Data
    parser.add_argument("--dataset_path", type=str, default="qwedsacf/competition_math")
    parser.add_argument("--prompt_template", type=str, default="boxed.prompt")
    
    # vLLM
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000")
    parser.add_argument("--max_tokens", type=int, default=1024)
    
    # Output
    parser.add_argument("--base_dir", type=str, default="runs")
    
    # Logging
    parser.add_argument("--wandb_project", type=str, default="math-grpo")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--disable_wandb", action="store_true")
    
    args = parser.parse_args()
    
    # Convert to TrainerConfig
    config = TrainerConfig(
        model_id=args.model_id,
        device=args.device,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lr=args.lr,
        n_grpo_steps=args.n_grpo_steps,
        n_prompts_per_step=args.n_prompts_per_step,
        group_size=args.group_size,
        epochs_per_step=args.epochs_per_step,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        use_std_norm=args.use_std_norm,
        use_kl_penalty=args.use_kl_penalty,
        use_clip=args.use_clip,
        beta=args.beta,
        clip_eps=args.clip_eps,
        use_length_penalty=args.use_length_penalty,
        use_prm=args.use_prm,
        expected_maximum_length=args.expected_maximum_length,
        overlong_cache=args.overlong_cache,
        process_reward_weight=args.process_reward_weight,
        use_token_level_loss=args.use_token_level_loss,
        prm_model_id=args.prm_model_id,
        prm_device=args.prm_device,
        prm_batch_size=args.prm_batch_size,
        dataset_path=args.dataset_path,
        prompt_template=args.prompt_template,
        vllm_url=args.vllm_url,
        max_tokens=args.max_tokens,
        base_dir=args.base_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        disable_wandb=args.disable_wandb,
    )
    
    return config


def main():
    config = parse_args()
    trainer = GRPOTrainer(config)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
