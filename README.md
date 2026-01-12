> This project is the final project for the PRML course: [Pattern Recognition and Machine Learning](https://github.com/OpenMOSS/FDU-PRML-2025Fall.git)

# PRML Course Final Project: RL Fine-Tuning for LLMs

> **Occam's Razor for RL Fine-Tuning: The Simple Design Is Efficient and Sufficient**

This is the companion code for the PRML (Pattern Recognition and Machine Learning) course final project. We study efficient RL fine-tuning of Large Language Models using GRPO with LoRA. We mainly focus on **3 questions**:

1. What LoRA capacity is sufficient?
2. Which RL techniques are truly effective?
3. What reward designs work best?

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
├── train.py                   # Unified training entry point
├── math_utils.py              # Answer verification
├── boxed.prompt               # Prompt template
├── pyproject.toml             # Project dependencies
├── src/
│   ├── __init__.py
│   ├── trainer.py             # GRPOTrainer class
│   ├── rewards.py             # Reward functions
│   ├── losses.py              # Loss functions
│   └── utils/
│       ├── __init__.py
│       ├── vllm_utils.py      # vLLM client
│       └── tokenization.py    # Tokenization utils
└── skywork_o1_prm_inference/  # PRM model (git submodule)
```

## Environment Setup

All dependencies are defined in `pyproject.toml`. Use [uv](https://docs.astral.sh/uv/) for installation:

```bash
# Install submodule (for PRM model)
git submodule update --init --recursive

# Install dependencies
uv sync
```

## Quick Start

### 1. Start vLLM Server
```bash
# Activate environment and start vLLM server
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
vllm serve Qwen/Qwen3-1.7B --enable-lora
```

### 2. Run Training (Best Configuration)
```bash
uv run train.py --disable_wandb
```
> **Note**: Please refer to `train.py` or the [Training Arguments](#training-arguments) section to add other necessary arguments (e.g., local model paths, dataset paths, specific device configurations) suitable for your environment.

## Experiments

### Experiment 1: LoRA Rank
```bash
uv run train.py --lora_r 1   # rank-1 (best efficiency)
uv run train.py --lora_r 4   # rank-4
uv run train.py --lora_r 16  # rank-16
```

### Experiment 2: RL Techniques
```bash
# Baseline with std normalization (recommended)
uv run train.py

# Without std normalization
uv run train.py --no_std_norm

# With KL penalty (degrades performance)
uv run train.py --use_kl_penalty

# With clipping (degrades performance)
uv run train.py --use_clip

# Token-level loss (no improvement)
uv run train.py --use_token_level_loss
```

### Experiment 3: Reward Design
```bash
# Baseline: outcome reward only (recommended)
uv run train.py

# With length penalty (degrades exploration)
uv run train.py --use_length_penalty

# With PRM (causes reward hacking)
uv run train.py --use_prm --process_reward_weight 0.5
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