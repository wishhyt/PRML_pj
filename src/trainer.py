"""
Unified GRPO Trainer for LLM Fine-tuning.

This trainer consolidates all experimental configurations from the paper
"Occam's Razor for RL Fine-Tuning: The Simple Design Is Efficient and Sufficient"

Default configuration uses the paper's recommended "best design":
- Standard deviation normalization: ON
- Simple outcome reward: ON  
- KL penalty: OFF
- Clipping: OFF
- Length penalty: OFF
- Process reward: OFF
- Token-level loss: OFF
"""

import os
import random
import time
import torch
import wandb
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from src.rewards import compute_outcome_reward, compute_length_penalty, compute_advantages
from src.losses import grpo_loss, grpo_loss_with_clip_kl, token_level_loss
from src.utils.vllm_utils import VLLMClient
from src.utils.tokenization import tokenize_prompt_and_output, get_response_log_probs
from math_utils import last_boxed_only_string, remove_boxed, is_equiv


@dataclass
class TrainerConfig:
    """Configuration for GRPO trainer."""
    
    # Model
    model_id: str = "Qwen/Qwen3-1.7B"
    device: str = "cuda:0"
    
    # LoRA
    lora_r: int = 1
    lora_alpha: int = 32
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "up_proj", "gate_proj", "down_proj",
    ])
    
    # Training
    lr: float = 9e-5
    n_grpo_steps: int = 50
    n_prompts_per_step: int = 32
    group_size: int = 8
    epochs_per_step: int = 1
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 128
    seed: int = 42
    
    # GRPO options (paper findings: keep these OFF for best results)
    use_std_norm: bool = True      # ON - recommended
    use_kl_penalty: bool = False   # OFF - degrades performance
    use_clip: bool = False         # OFF - degrades performance
    beta: float = 0.04             # KL coefficient (if use_kl_penalty=True)
    clip_eps: float = 0.2          # Clip epsilon (if use_clip=True)
    
    # Reward options (paper findings: keep these OFF for best results)
    use_length_penalty: bool = False  # OFF - degrades exploration
    use_prm: bool = False             # OFF - causes reward hacking
    expected_maximum_length: int = 1024
    overlong_cache: int = 512
    process_reward_weight: float = 0.5
    
    # Loss options
    use_token_level_loss: bool = False  # OFF - no improvement
    
    # PRM settings (only if use_prm=True)
    prm_model_id: str = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
    prm_device: str = "auto"
    prm_batch_size: int = 16
    prm_step_token: str = "\n"
    
    # Data
    dataset_path: str = "qwedsacf/competition_math"
    prompt_template: str = "boxed.prompt"
    
    # vLLM
    vllm_url: str = "http://localhost:8000"
    max_tokens: int = 1024
    
    # Output
    base_dir: str = "runs"
    
    # Logging
    wandb_project: str = "math-grpo"
    wandb_run_name: Optional[str] = None
    disable_wandb: bool = False


class GRPOTrainer:
    """
    Unified GRPO trainer with configurable options.
    
    Uses the paper's recommended configuration by default.
    """
    
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.run_name = None
        self.vllm_client = None
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.template = None
        
        # PRM components (only initialized if needed)
        self.prm_model = None
        self.prm_tokenizer = None
        
    def setup(self):
        """Initialize all components."""
        cfg = self.config
        
        # Set seeds
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        
        # Print configuration
        self._print_config()
        
        # Setup vLLM
        self.vllm_client = VLLMClient(cfg.vllm_url)
        if not self.vllm_client.check_connection():
            raise RuntimeError("Could not connect to vLLM server")
        
        # Setup run directory
        self._setup_run_dir()
        
        # Initialize wandb
        self._setup_wandb()
        
        # Load data
        self._load_data()
        
        # Load model
        self._load_model()
        
        # Load PRM if needed
        if cfg.use_prm:
            self._load_prm()
    
    def _print_config(self):
        cfg = self.config
        print("=" * 60)
        print("GRPO Training Configuration")
        print("=" * 60)
        print(f"Model: {cfg.model_id}")
        print(f"LoRA rank: {cfg.lora_r}")
        print(f"Learning rate: {cfg.lr}")
        print()
        print("Techniques (paper-recommended defaults):")
        print(f"  std normalization: {'ON' if cfg.use_std_norm else 'OFF'}")
        print(f"  KL penalty: {'ON' if cfg.use_kl_penalty else 'OFF'}")
        print(f"  Clipping: {'ON' if cfg.use_clip else 'OFF'}")
        print(f"  Length penalty: {'ON' if cfg.use_length_penalty else 'OFF'}")
        print(f"  Process reward: {'ON' if cfg.use_prm else 'OFF'}")
        print(f"  Token-level loss: {'ON' if cfg.use_token_level_loss else 'OFF'}")
        print("=" * 60)
    
    def _setup_run_dir(self):
        cfg = self.config
        os.makedirs(cfg.base_dir, exist_ok=True)
        i = 1
        while os.path.exists(f"{cfg.base_dir}/{i}"):
            i += 1
        self.run_name = f"{cfg.base_dir}/{i}"
        os.makedirs(self.run_name)
        print(f"Run directory: {self.run_name}")
    
    def _setup_wandb(self):
        cfg = self.config
        if not cfg.disable_wandb:
            run_name = cfg.wandb_run_name or f"{cfg.model_id}_lr{cfg.lr:.1e}_r{cfg.lora_r}"
            wandb.init(
                project=cfg.wandb_project,
                name=run_name,
                config=vars(cfg),
                dir=self.run_name,
            )
            wandb.config.update({"run_dir": self.run_name})
    
    def _load_data(self):
        cfg = self.config
        print("Loading dataset...")
        self.train_dataset = load_dataset(cfg.dataset_path, split="train[:7500]")
        self.val_dataset = load_dataset(cfg.dataset_path, split="train[-5000:]")
        
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        with open(cfg.prompt_template, "r", encoding="utf-8") as f:
            self.template = f.read().strip()
        
        def process_data(example):
            with_template = self.template.replace("{question}", example["problem"])
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": with_template}],
                tokenize=False,
                add_generation_prompt=True,
            )
            answer = remove_boxed(last_boxed_only_string(example["solution"]))
            return {"prompt": prompt, "answer": answer, "problem": example["problem"]}
        
        self.train_dataset = self.train_dataset.map(process_data)
        self.val_dataset = self.val_dataset.map(process_data)
        print(f"Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")
    
    def _load_model(self):
        cfg = self.config
        print("Loading model...")
        model_kwargs = dict(
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            use_cache=False,
            device_map=cfg.device,
        )
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **model_kwargs)
        
        peft_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.lora_target_modules,
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
    
    def _load_prm(self):
        cfg = self.config
        print("Loading PRM model...")
        from skywork_o1_prm_inference.model_utils.prm_model import PRM_MODEL
        from skywork_o1_prm_inference.model_utils.io_utils import (
            prepare_input, prepare_batch_input_for_model, derive_step_rewards
        )
        
        self.prm_tokenizer = AutoTokenizer.from_pretrained(cfg.prm_model_id, trust_remote_code=True)
        if self.prm_tokenizer.pad_token_id is None:
            self.prm_tokenizer.pad_token_id = self.prm_tokenizer.eos_token_id
        self.prm_model = PRM_MODEL.from_pretrained(cfg.prm_model_id, device_map=cfg.prm_device).eval()
        
        # Store helper functions
        self._prm_prepare_input = prepare_input
        self._prm_prepare_batch = prepare_batch_input_for_model
        self._prm_derive_rewards = derive_step_rewards
    
    def save_lora(self, step: int) -> str:
        """Save LoRA weights and return the path."""
        lora_name = f"{self.run_name}/step={step}"
        if not os.path.exists(lora_name):
            self.model.save_pretrained(lora_name)
        return lora_name
    
    def evaluate(self, step: int) -> float:
        """Run evaluation and return accuracy."""
        cfg = self.config
        lora_name = self.save_lora(step)
        self.vllm_client.load_lora(lora_name)
        
        val_prompts = self.val_dataset[:1000]["prompt"]
        
        eval_start = time.time()
        outputs = self.vllm_client.generate(
            val_prompts, lora_name,
            max_tokens=cfg.max_tokens,
            temperature=0,
        )
        eval_time = time.time() - eval_start
        
        correct = 0
        for i, output in enumerate(outputs):
            correct_answer = self.val_dataset[i]["answer"]
            generated_answer = remove_boxed(last_boxed_only_string(output))
            if generated_answer is not None and is_equiv(generated_answer, correct_answer):
                correct += 1
        
        accuracy = correct / len(outputs)
        print(f"Step {step}: {correct}/{len(outputs)} ({accuracy:.2%})")
        
        if not cfg.disable_wandb:
            wandb.log({
                "eval/accuracy": accuracy,
                "eval/correct": correct,
                "eval/total": len(outputs),
                "eval/time": eval_time,
            }, step=step)
        
        return accuracy
    
    def train_step(self, step: int):
        """Execute one GRPO training step."""
        cfg = self.config
        step_start = time.time()
        
        # Sample batch
        indices = random.sample(range(len(self.train_dataset)), cfg.n_prompts_per_step)
        batch = self.train_dataset[indices]
        
        # Save and load LoRA for generation
        lora_name = self.save_lora(step)
        self.vllm_client.load_lora(lora_name)
        
        # Generate rollouts
        gen_start = time.time()
        outputs = self.vllm_client.generate(
            batch["prompt"], lora_name,
            max_tokens=cfg.max_tokens,
            temperature=1,
            n=cfg.group_size,
        )
        gen_time = time.time() - gen_start
        
        # Compute rewards
        rewards = compute_outcome_reward(outputs, batch["answer"], cfg.group_size)
        
        if cfg.use_length_penalty:
            length_penalties = compute_length_penalty(
                outputs, self.tokenizer,
                cfg.expected_maximum_length, cfg.overlong_cache
            ).reshape(cfg.n_prompts_per_step, cfg.group_size)
            rewards = rewards + length_penalties
        
        if cfg.use_prm:
            from src.rewards import compute_process_reward
            problems_expanded = [x for x in batch["problem"] for _ in range(cfg.group_size)]
            # Format for PRM
            outputs_formatted = [o.replace(". ", ".\n").replace("。", "。\n") for o in outputs]
            process_rewards = compute_process_reward(
                problems_expanded, outputs_formatted,
                self.prm_model, self.prm_tokenizer,
                cfg.prm_batch_size, cfg.prm_step_token,
                self._prm_prepare_input, self._prm_prepare_batch, self._prm_derive_rewards,
            ).reshape(cfg.n_prompts_per_step, cfg.group_size)
            
            w = cfg.process_reward_weight
            rewards = (1 - w) * rewards + w * process_rewards
        
        # Compute advantages
        advantages = compute_advantages(rewards, use_std_norm=cfg.use_std_norm)
        
        # Stats
        train_accuracy = compute_outcome_reward(outputs, batch["answer"], cfg.group_size).mean().item()
        
        # Tokenize
        prompts_expanded = [x for x in batch["prompt"] for _ in range(cfg.group_size)]
        data = tokenize_prompt_and_output(prompts_expanded, outputs, self.tokenizer)
        input_ids = data["input_ids"].to(cfg.device)
        labels = data["labels"].to(cfg.device)
        response_mask = data["response_mask"].to(cfg.device)
        
        # Token-level loss needs group output lengths
        group_output_lens = None
        if cfg.use_token_level_loss:
            group_output_lens = response_mask.reshape(
                cfg.n_prompts_per_step, cfg.group_size, -1
            ).sum(dim=-1).reshape(-1).to(cfg.device)
        
        # Compute old log probs
        with torch.inference_mode():
            old_logprobs_all = []
            for b in range(len(input_ids) // cfg.micro_batch_size):
                idx = b * cfg.micro_batch_size
                end = idx + cfg.micro_batch_size
                old_logprobs_all.append(
                    get_response_log_probs(self.model, input_ids[idx:end], labels[idx:end]).detach()
                )
            old_logprobs_all = torch.cat(old_logprobs_all, dim=0)
        
        # Training loop
        all_metrics = {"approx_kl": [], "policy_ratio": [], "grad_norm": []}
        
        train_start = time.time()
        for epoch in range(cfg.epochs_per_step):
            for b in tqdm(
                range(len(input_ids) // cfg.micro_batch_size),
                desc=f"Step {step+1}/{cfg.n_grpo_steps}",
            ):
                idx = b * cfg.micro_batch_size
                end = idx + cfg.micro_batch_size
                
                x = input_ids[idx:end]
                y = labels[idx:end]
                mask = response_mask[idx:end]
                adv = advantages[idx:end].to(cfg.device)
                old_lp = old_logprobs_all[idx:end]
                
                policy_lp = get_response_log_probs(self.model, x, y)
                
                # Select loss function
                if cfg.use_kl_penalty or cfg.use_clip:
                    loss, metrics = grpo_loss_with_clip_kl(
                        policy_lp, old_lp, adv, mask,
                        clip_eps=cfg.clip_eps if cfg.use_clip else 1e9,
                        beta=cfg.beta if cfg.use_kl_penalty else 0,
                    )
                elif cfg.use_token_level_loss:
                    gol = group_output_lens[idx:end] if group_output_lens is not None else None
                    loss, metrics = token_level_loss(
                        policy_lp, old_lp, adv, mask, gol, cfg.group_size
                    )
                else:
                    loss, metrics = grpo_loss(policy_lp, old_lp, adv, mask)
                
                loss = loss / cfg.gradient_accumulation_steps
                loss.backward()
                
                all_metrics["approx_kl"].append(metrics["approx_kl"])
                all_metrics["policy_ratio"].append(metrics["policy_ratio"])
                
                if (b + 1) % cfg.gradient_accumulation_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    all_metrics["grad_norm"].append(grad_norm.item())
                    self.optimizer.step()
                    self.optimizer.zero_grad()
        
        train_time = time.time() - train_start
        step_time = time.time() - step_start
        
        # Log
        mean_kl = sum(all_metrics["approx_kl"]) / len(all_metrics["approx_kl"]) if all_metrics["approx_kl"] else 0
        mean_grad = sum(all_metrics["grad_norm"]) / len(all_metrics["grad_norm"]) if all_metrics["grad_norm"] else 0
        avg_gen_len = int(response_mask.sum(dim=-1).float().mean().item())
        
        print(f"Step {step+1}/{cfg.n_grpo_steps} | Acc: {train_accuracy:.2%} | KL: {mean_kl:.4f} | Time: {step_time:.1f}s")
        
        if not cfg.disable_wandb:
            wandb.log({
                "train/accuracy": train_accuracy,
                "train/reward_mean": rewards.mean().item(),
                "train/approx_kl": mean_kl,
                "train/grad_norm": mean_grad,
                "train/avg_gen_length": avg_gen_len,
                "time/step": step_time,
                "time/generation": gen_time,
                "time/training": train_time,
            }, step=step+1)
    
    def train(self):
        """Run full training loop."""
        cfg = self.config
        
        print("\nStarting initial evaluation...")
        self.model.to(cfg.device)
        self.evaluate(0)
        self.model.train()
        
        for step in range(cfg.n_grpo_steps):
            self.train_step(step)
            
            if (step + 1) % 5 == 0:
                self.evaluate(step + 1)
        
        if not cfg.disable_wandb:
            wandb.finish()
        
        print("\nTraining complete!")
