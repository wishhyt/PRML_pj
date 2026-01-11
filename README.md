> This project is the final project for the PRML course: [Pattern Recognition and Machine Learning](https://github.com/OpenMOSS/FDU-PRML-2025Fall.git)

# PRML Course Final Project: RL Fine-Tuning for LLMs

> **Occam's Razor for RL Fine-Tuning: The Simple Design Is Efficient and Sufficient**

This is the companion code for the PRML (Pattern Recognition and Machine Learning) course final project. We study efficient RL fine-tuning of Large Language Models using GRPO with LoRA.

## Key Findings

| Technique | Effect | Recommendation |
|-----------|--------|----------------|
| **Std Normalization** | ✅ Improves performance | **Use** |
| **Outcome Reward** | ✅ Simple and effective | **Use** |
| KL Penalty | ❌ Degrades performance | Avoid |
| PPO Clipping | ❌ Degrades performance | Avoid |
| Token-level Loss | ❌ No improvement | Avoid |
| Length Penalty | ❌ Degrades exploration | Avoid |
| Process Reward (PRM) | ❌ Causes reward hacking | Avoid |

**Conclusion**: In minimal settings, additional complexity is often unnecessary. **rank-1 LoRA + simple GRPO** achieves strong performance.

## Project Structure

```
├── train.py              # Unified training entry point
├── src/
│   ├── trainer.py        # GRPOTrainer class
│   ├── rewards.py        # Reward functions
│   ├── losses.py         # Loss functions
│   └── utils/
│       ├── vllm_utils.py    # vLLM client
│       └── tokenization.py  # Tokenization utils
├── math_utils.py         # Answer verification
├── boxed.prompt          # Prompt template
└── skywork-o1-prm-inference/  # PRM model
```

## Environment Setup

### Requirements

- Python >= 3.13
- PyTorch >= 2.8.0
- Transformers >= 4.57.1
- PEFT >= 0.17.1
- vLLM == 0.10.2
- Datasets >= 4.2.0
- WandB >= 0.22.2

### Installation

```bash
# install submodule
git submodule update --init --recursive

# Using uv (recommended)
uv sync
source .venv/bin/activate

# Or using pip
pip install -e .
```

## Quick Start

### 1. Start vLLM Server
```bash
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
vllm serve Qwen/Qwen3-1.7B --enable-lora
```

### 2. Run Training (Best Configuration)
```bash
python train.py --disable_wandb
```

## Experiments

### Experiment 1: LoRA Rank
```bash
python train.py --lora_r 1   # rank-1 (best efficiency)
python train.py --lora_r 4   # rank-4
python train.py --lora_r 16  # rank-16
```

### Experiment 2: RL Techniques
```bash
# Baseline (recommended)
python train.py

# With KL penalty (degrades performance)
python train.py --use_kl_penalty

# With clipping (degrades performance)
python train.py --use_clip

# Token-level loss (no improvement)
python train.py --use_token_level_loss
```

### Experiment 3: Reward Design
```bash
# Baseline: outcome reward only (recommended)
python train.py

# With length penalty (degrades exploration)
python train.py --use_length_penalty

# With PRM (causes reward hacking)
python train.py --use_prm --process_reward_weight 0.5
```

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_id` | Qwen/Qwen3-1.7B | Base model |
| `--lora_r` | 1 | LoRA rank |
| `--lr` | 9e-5 | Learning rate |
| `--n_grpo_steps` | 50 | Training steps |
| `--group_size` | 8 | Rollouts per prompt |
| `--use_std_norm` | True | Std normalization |
| `--use_kl_penalty` | False | KL penalty |
| `--use_clip` | False | PPO clipping |
| `--use_length_penalty` | False | Length penalty |
| `--use_prm` | False | Process reward |

## Acknowledgement

This project is inspired by [LoRA without regret](https://thinkingmachines.ai/blog/lora/) and its reproduction [code](https://github.com/michaelbzhu/lora-without-regret/)
