# Thinnka Podsmith

Provision Runpod Pods for GRPO or SFT fine-tuning with the Open R1 repo. This project:
- checks if a Hugging Face model is gated
- estimates model size to pick a GPU with enough VRAM
- prefers L40S, then A100 SXM, then H100 SXM
- creates a Runpod Pod with a target Docker image
- SSHes into the Pod to run Open R1 GRPO or SFT training
- pushes the result to the Hugging Face Hub using your `HF_TOKEN`
- streams training progress to an async webhook and prints remote output locally

## Prereqs
- Windows (or WSL) with Python 3.11
- Runpod API key
- Hugging Face token with write access
- SSH key pair added to Runpod account settings
- Pod image that allows SSH on port 22

## Setup (Windows)
1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Copy `example.env` to `.env` and fill in values.

## Environment variables
- `RUNPOD_API_KEY` (required)
- `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` (required)
- `PROGRESS_WEBHOOK_URL` (optional, async progress events)
- `DISCORD_WEBHOOK_URL` (optional, Discord progress logs)
- `SSH_PUBLIC_KEY` or `RUNPOD_SSH_PUBLIC_KEY` (recommended)
- `SSH_PRIVATE_KEY_PATH` (optional, defaults to `~/.ssh/id_ed25519`)
- `WANDB_API_KEY` (optional if you pass `--report-to wandb`)

## .env loading
The runner loads `.env` automatically from the project directory.
You can override the path with `--env-file`.
`.env` values take precedence over user/system environment variables.

## Hugging Face token
Uploading to the Hub requires a **write** token. Read-only tokens will fail with 401.
The runner uploads the token to the pod and writes it to `$HF_HOME/token` so
training can authenticate reliably.

## SSH note
The script waits for a public IP and a mapped port for `22/tcp`.
If you use a custom image, ensure `sshd` is running and port 22 is exposed.

## Progress webhook
Set `PROGRESS_WEBHOOK_URL` to receive JSON updates. Payload includes:
- `event`
- `message`
- `timestamp`
- `stage` (setup or train when streaming logs)
- `project`
- `repo_id`

## Discord webhook
Set `DISCORD_WEBHOOK_URL` or pass `--discord-webhook-url` to send the same events to Discord.
Each event is sent as a text message (errors include recent output).

## Logging
Remote stdout/stderr lines are printed to the local console with a `[setup]` or `[train]` prefix. Webhook and Discord logs remain throttled to avoid spam.

## Quick start
Runs the test case model and creates a Pod with the default image (GRPO, no answer tags):
- `python thinnka_runner.py --repo-id unsloth/gemma-2b --gpu-count 1`

Run supervised fine-tuning (SFT):
- `python thinnka_runner.py --repo-id unsloth/gemma-2b --gpu-count 1 --sft`

Use the custom image if you build and push it (see below):
- `python thinnka_runner.py --repo-id unsloth/gemma-2b --gpu-count 1 --image reeeon/thinnka:latest`
- Add `--skip-setup` when using the custom image to avoid reinstalling Open R1.

Enable Weights and Biases logging:
- `python thinnka_runner.py --repo-id unsloth/gemma-2b --gpu-count 1 --report-to wandb`

Debug remote shell commands:
- `python thinnka_runner.py --repo-id unsloth/gemma-2b --gpu-count 1 --debug-remote`

## Attention implementation
By default the runner uses `sdpa` to avoid FlashAttention ABI issues on some images.
If you want FlashAttention, pass:
- `--attn-implementation flash_attention_2`

When `flash_attention_2` is selected, the setup script installs `flash-attn`.

## Accelerate config
The runner generates an Accelerate config that matches your `--gpu-count`.
You can switch ZeRO stage with:
- `--deepspeed-stage 2` or `--deepspeed-stage 3`

## Storage
The persistent volume defaults to 300 GB. Override it with:
- `--volume-gb 500`

## SFT mode
Pass `--sft` to run Open R1's supervised fine-tuning script (`src/open_r1/sft.py`).
The runner sets `max_seq_length` to `max_prompt_length + max_completion_length` so the existing CLI flags still apply.
GRPO-only options (reward functions, `--num-generations`, and vLLM) are ignored in SFT mode.
SFT runs use QLoRA (4-bit + LoRA) rather than full fine-tuning.
Because QLoRA uses a device map, the runner forces ZeRO-2 when `--sft` is set (ZeRO-3 is incompatible).
The setup step installs `peft` and `bitsandbytes` and verifies they import successfully.

## GRPO generation backend (GRPO only)
GRPO is an online method: it must generate completions during training to compute rewards.
By default the runner uses vLLM for faster generation. If you hit vLLM/NCCL issues,
you can switch to the transformers generation path:
- `--no-vllm` (slower, but avoids vLLM-specific hangs)

## GRPO generations (GRPO only)
`num_generations` must divide the effective batch size. The runner auto-adjusts
if needed. Override with:
- `--num-generations 4`

## Chat templates
Some base models (like `unsloth/gemma-2b`) do not ship a chat template.
The runner detects this and injects a simple fallback template.
You can override it with:
- `--chat-template path\to\template.jinja`

## Failure cleanup
If the run fails after a Pod is created, the script automatically stops and attempts to terminate the Pod to avoid extra charges.

## Custom image (Open R1 preinstalled)
This repo includes a Dockerfile that builds `reeeon/thinnka:latest`.

Build and push:
- `docker build -t reeeon/thinnka:latest .`
- `docker push reeeon/thinnka:latest`

Notes:
- Building can take time because Open R1 installs vLLM and flash-attn.
- Open R1 recommends CUDA 12.4 and PyTorch 2.6.0; the base image here is CUDA 12.8.1.
- If you build on macOS, use `--platform linux/amd64`.

## How it works
1. Fetches `model_info` from the Hub and stops if the model is gated.
2. Sums model file sizes to estimate required VRAM.
3. Queries Runpod GPU types and picks from: L40S, A100 SXM, H100 SXM.
4. Creates a Pod with your selected GPU count (1, 2, 4, 6, 8).
5. Waits for SSH on port 22, then installs Open R1 if needed.
6. Generates a GRPO or SFT config and launches training.
7. Uses `HF_TOKEN` to push to the Hub.
8. Streams progress lines to `PROGRESS_WEBHOOK_URL` and optionally `DISCORD_WEBHOOK_URL`.

## Reasoning tag behavior (GRPO only)
By default, the config uses `<think> ... </think>` and **no answer tags** (accuracy reward only).
If you enable answer tags, the script uses Open R1's default format rewards.
If you change the reasoning or answer tags, the script falls back to safer rewards
because Open R1 format rewards are hardcoded to the default tags.
Use `--reasoning-tag`, `--reasoning-end-tag`, `--answer-tag`, and `--answer-end-tag` to customize.

Enable answer tags:
- `--answer-tag "<answer>" --answer-end-tag "</answer>"`

## References
- Runpod docs: https://docs.runpod.io/overview
- Runpod GraphQL Pods: https://docs.runpod.io/sdks/graphql/manage-pods
- Hugging Face Hub Python: https://huggingface.co/docs/huggingface_hub/index
- Open R1 repo: https://github.com/huggingface/open-r1
