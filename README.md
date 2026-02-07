<div align="center">

# 🚀 Thinnka Podsmith

**Automate Runpod Pod provisioning & train Open R1 models with zero friction**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue?style=for-the-badge&logo=python)](https://www.python.org/downloads/release/python-3110/)
[![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## ✨ What it does

| Feature | Description |
|----------|-------------|
| 🔐 **Gated check** | Detects and blocks gated Hugging Face models upfront |
| 📊 **Smart VRAM** | Estimates model size → auto-selects optimal GPU |
| 🎯 **GPU priority** | Prefers L40S → A100 SXM → H100 SXM |
| 🐳 **Pod provisioning** | Creates Runpod pods with your Docker image |
| 🧠 **Training modes** | Supports both GRPO and SFT fine-tuning |
| 📤 **Auto upload** | Pushes trained models to Hugging Face Hub |
| 📡 **Progress streaming** | Real-time webhooks + Discord integration |

---

## 📦 Quick Start

```bash
# Install
pip install -r requirements.txt

# Configure
cp example.env .env
# Edit .env with your API keys

# Run GRPO training
python thinnka_runner.py --repo-id unsloth/gemma-2b --gpu-count 1

# Run SFT training
python thinnka_runner.py --repo-id unsloth/gemma-2b --gpu-count 1 --sft
```

---

## 🛠️ Prerequisites

| Requirement | Details |
|-------------|---------|
| **OS** | Windows (or WSL) with Python 3.11 |
| **Runpod** | API key from Runpod console |
| **Hugging Face** | Token with **write** access (read-only fails) |
| **SSH keys** | Key pair added to Runpod settings |
| **Docker** | Image exposing port 22 for SSH |

---

## ⚙️ Configuration

### Environment Variables

Create `.env` from `example.env`:

```bash
RUNPOD_API_KEY=your_runpod_key              # Required
HF_TOKEN=your_hf_token                    # Required (write access)
PROGRESS_WEBHOOK_URL=https://...            # Optional
DISCORD_WEBHOOK_URL=https://discord.com/...  # Optional
SSH_PUBLIC_KEY=ssh-ed25519 AAAA...         # Recommended
SSH_PRIVATE_KEY_PATH=~/.ssh/id_ed25519      # Optional
WANDB_API_KEY=your_wandb_key             # Optional
```

> **Note:** `.env` values override system environment variables.

### Key Behaviors

| Aspect | Behavior |
|---------|----------|
| **HF Token** | Uploaded to pod at `$HF_HOME/token` for reliable auth |
| **SSH** | Waits for public IP + port 22 mapping |
| **Webhook** | JSON payload with `event`, `message`, `timestamp`, `stage`, `project`, `repo_id` |
| **Discord** | Same events as text messages (errors include recent output) |
| **Logging** | Remote output streamed locally with `[setup]`/`[train]` prefix |

---

## 🎯 Common Use Cases

### Force specific GPU

```bash
python thinnka_runner.py --repo-id unsloth/gemma-2b --gpu-count 1 --gpu-type "L40S"
```

### Use custom Docker image

```bash
# Build & push
docker build -t reeeon/thinnka:latest .
docker push reeeon/thinnka:latest

# Run with custom image
python thinnka_runner.py --repo-id unsloth/gemma-2b --gpu-count 1 \
  --image reeeon/thinnka:latest --skip-setup
```

### Enable W&B logging

```bash
python thinnka_runner.py --repo-id unsloth/gemma-2b --gpu-count 1 --report-to wandb
```

### Debug remote commands

```bash
python thinnka_runner.py --repo-id unsloth/gemma-2b --gpu-count 1 --debug-remote
```

---

## 🧠 Training Modes

### GRPO (Default)

- **Online RL** with vLLM generation (faster)
- Reward functions: accuracy, format, tag_count (default tags)
- Fallback to transformers: `--no-vllm`

### SFT (Supervised Fine-Tuning)

```bash
python thinnka_runner.py --repo-id unsloth/gemma-2b --gpu-count 1 --sft
```

| Feature | GRPO | SFT |
|---------|-------|-----|
| **Method** | Online RL | Supervised |
| **Efficiency** | QLoRA (4-bit + LoRA) | QLoRA (4-bit + LoRA) |
| **ZeRO** | Stage 3 | Stage 2 (QLoRA incompatible with ZeRO-3) |
| **vLLM** | Yes | No |
| **Large models** | Use `--shard-model` | Use `--shard-model` to disable QLoRA |

> **📌 Sharded models:** Pass `--shard-model` for large models to disable QLoRA and use ZeRO-3 sharding instead.

### Model Merging (SFT + QLoRA Only)

By default, SFT saves **LoRA adapters** (10-100MB). Use `--merge-model` to automatically merge into base model after training:

```bash
# Train and merge automatically
python thinnka_runner.py --repo-id unsloth/gemma-2b --gpu-count 1 --sft --merge-model
```

| Feature | Without --merge-model | With --merge-model |
|---------|---------------------|-------------------|
| **Uploads** | LoRA adapters only | Merged full model |
| **Size** | ~10-100 MB | ~Model size (GBs) |
| **Usage** | Load with `PeftModel` | Load directly |
| **Flexibility** | Apply to any base | Fixed to trained base |

> **⚠️ Requirements:** Only works with SFT + QLoRA (default). Incompatible with `--shard-model` or GRPO mode.

---

## 🎛️ Advanced Options

### Attention Implementation

| Option | Use Case |
|---------|----------|
| `sdpa` (default) | Avoids FlashAttention ABI issues |
| `flash_attention_2` | Faster, requires install |

```bash
--attn-implementation flash_attention_2
```

### ZeRO Stage

```bash
--deepspeed-stage 2  # or 3
```

### Storage

```bash
--volume-gb 500           # Persistent volume
--container-disk-gb 200   # Container disk
```

### Speed Controls

```bash
--max-steps 500                     # Limit training steps
--dataset-fraction 0.1              # Subsample dataset
--dataset-seed 123                   # Fixed seed for reproducibility
```

### Chat Templates

Auto-detects and injects missing templates (e.g., `unsloth/gemma-2b`):

```bash
--chat-template path/to/template.jinja
```

### Custom Dependencies

Install additional pip packages required by specific models:

```bash
# Single package
--custom-dependencies "protobuf"

# Multiple packages
--custom-dependencies "package1,package2,package3"

# Specific version
--custom-dependencies "torchvision==0.20.0"
```

> **⚠️ Use case:** Some models (e.g., `openbmb/MiniCPM-o-4_5`) require specific dependencies for custom code execution. Combine with `--trust-remote-code` when needed.

### Reasoning Tags (GRPO)

Default: `</think>` tags only (accuracy reward)

```bash
# Enable answer tags
--answer-tag "<answer>" --answer-end-tag "</answer>"

# Custom tags
--reasoning-tag "<think>" --reasoning-end-tag "</think>"
```

---

## 🔧 Transformers Version Control

**Prevents MoE weight-conversion bugs** (e.g., Qwen3-Next in Transformers v5.0 dev)

### Automatic Detection

| Priority | Source |
|----------|---------|
| 1️⃣ | `--transformers-version` flag |
| 2️⃣ | Model's `config.json` → `transformers_version` field |
| 3️⃣ | Default: `4.57.6` |

### Override Options

```bash
# Force git HEAD (dev version)
--transformers-from-git

# Explicit version
--transformers-version 4.57.6
```

> **✅ Result:** Auto-pins to stable versions, prevents ZeRO-3 MoE errors while allowing opt-in to dev for testing.

---

## 📊 Pipeline Flow

```mermaid
graph LR
    A[Check Gated] --> B[Estimate VRAM]
    B --> C[Select GPU]
    C --> D[Create Pod]
    D --> E[SSH Connect]
    E --> F[Install Open R1]
    F --> G[Generate Config]
    G --> H[Launch Training]
    H --> I[Push to Hub]
    I --> J[Stream Progress]
```

1. 📥 Fetches `model_info` from Hub → stops if gated
2. 📐 Sums model file sizes → estimates VRAM
3. 🔍 Queries Runpod GPU types → picks L40S/A100/H100
4. 🚀 Creates Pod (1/2/4/6/8 GPUs)
5. 🔌 Waits for SSH port 22
6. 📦 Installs Open R1 (if needed)
7. ⚙️ Generates GRPO/SFT config
8. ▶️ Launches training
9. 📤 Pushes to Hub via `HF_TOKEN`
10. 📡 Streams to webhook + Discord

---

## 🏗️ Custom Docker Image

### SFT-Focused Build

Excludes vLLM/flash-attn, includes QLoRA deps:

```bash
# Build
docker build -t thinnka-sft .

# Run locally
docker run --gpus all -it --rm -e HF_TOKEN=your_token thinnka-sft bash

# Inside container
cd /opt/open-r1
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
  src/open_r1/sft.py --config recipes/OpenR1-Distill-7B/sft/config_distill.yaml
```

### Notes

| Aspect | Detail |
|--------|--------|
| **Base image** | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| **CUDA** | 12.8.1 (Open R1 recommends 12.4) |
| **PyTorch** | 2.6.0 |
| **macOS build** | Use `--platform linux/amd64` |

---

## 📋 Changes

### February 7, 2026

| Change | Description |
|--------|-------------|
| ✨ | Auto-detect Transformers version from model config |
| 🔒 | Default pinned to `4.57.6` (prevents MoE bugs) |
| 🚩 | Added `--transformers-from-git` flag |
| 🎯 | Added `--transformers-version` flag |
| 🤝 | Added `--merge-model` flag for SFT + QLoRA |
| 📦 | Added `--custom-dependencies` flag for pip packages |
| 🔐 | Added `--trust-remote-code` flag for custom model code |
| 🐛 | Fixed Qwen3-Next ZeRO-3 conversion errors |

---

## 📦 Uploaded Models

> **⚠️ Mockup data for demonstration only**

| Model | Hub ID | Type | Transformers | Status |
|--------|----------|------|--------------|----------|
| Qwen3-Next-7B | `username/qwen3-next-7b-grpo` | GRPO | 4.57.6 | 🚧 Mockup |
| Qwen3-Next-7B | `username/qwen3-next-7b-sft` | SFT | 4.57.6 | 🚧 Mockup |
| Gemma-2B | `username/gemma-2b-finetuned` | SFT | 4.57.6 | 🚧 Mockup |

**Usage:**
```bash
python thinnka_runner.py --repo-id username/qwen3-next-7b-grpo --gpu-count 1
python thinnka_runner.py --repo-id username/qwen3-next-7b-sft --gpu-count 1 --sft
```

---

## 📚 References

| Resource | Link |
|----------|------|
| **Runpod Docs** | https://docs.runpod.io/overview |
| **Runpod GraphQL** | https://docs.runpod.io/sdks/graphql/manage-pods |
| **Hugging Face Hub** | https://huggingface.co/docs/huggingface_hub |
| **Open R1 Repo** | https://github.com/huggingface/open-r1 |

---

<div align="center">

**Built with ❤️ for the ML community**

</div>
