#!/usr/bin/env python3
"""
Thinnka Podsmith: Runpod Pod provisioning + Open R1 GRPO runner.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
import paramiko
import yaml
from dotenv import find_dotenv, load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
try:
    from discord_webhook_async import DiscordWebhook
except ImportError:
    DiscordWebhook = None

PROJECT_NAME = "Thinnka Podsmith"
RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"
RUNPOD_REST_URL = "https://rest.runpod.io/v1"

DEFAULT_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
CUSTOM_IMAGE = "reeeon/thinnka:latest"

ALLOWED_GPU_COUNTS = {1, 2, 4, 6, 8}
WEIGHT_EXTENSIONS = {".safetensors", ".bin", ".pt", ".pth", ".gguf", ".ckpt"}

GPU_PRIORITY = [
    ("L40S", lambda name: "l40s" in name),
    ("A100 SXM", lambda name: "a100" in name and "sxm" in name),
    ("H100 SXM", lambda name: "h100" in name and "sxm" in name),
]

DEFAULT_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ message['content'] + '\\n' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ 'User: ' + message['content'] + '\\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ 'Assistant: ' + message['content'] + '\\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ 'Assistant: ' }}"
    "{% endif %}"
)

DISCORD_MAX_CONTENT = 2000


def _truncate_text(text: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return text[: max_len - 3] + "..."


def format_discord_message(payload: Dict[str, Any]) -> str:
    project = str(payload.get("project") or "").strip()
    repo_id = str(payload.get("repo_id") or "").strip()
    event = str(payload.get("event") or "event").strip()
    message = payload.get("message")
    stage = payload.get("stage")
    tail = payload.get("tail")

    header_parts = []
    if project and repo_id:
        header_parts.append(f"[{project} | {repo_id}]")
    elif project:
        header_parts.append(f"[{project}]")
    elif repo_id:
        header_parts.append(f"[{repo_id}]")
    header_parts.append(event)
    if stage:
        header_parts.append(f"({stage})")
    header = " ".join(header_parts).strip()
    if message:
        header = f"{header}: {str(message).strip()}"
    header = _truncate_text(header, DISCORD_MAX_CONTENT)

    if tail:
        tail_text = str(tail).strip().replace("```", "'''")
        if tail_text:
            max_tail = DISCORD_MAX_CONTENT - len(header) - (len("\n```") + len("```"))
            if max_tail > 0:
                tail_text = _truncate_text(tail_text, max_tail)
                return f"{header}\n```{tail_text}```"

    return header


class ProgressReporter:
    def __init__(
        self,
        url: Optional[str],
        base_payload: Optional[Dict[str, Any]] = None,
        discord_url: Optional[str] = None,
    ) -> None:
        self.url = url
        self.discord_url = discord_url
        self.base_payload = base_payload or {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue: Optional[asyncio.Queue] = None
        self._thread: Optional[threading.Thread] = None
        if self.url or self.discord_url:
            self._start()

    def _start(self) -> None:
        def runner() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            queue: asyncio.Queue = asyncio.Queue()
            self._loop = loop
            self._queue = queue
            loop.create_task(self._worker())
            loop.run_forever()

        self._thread = threading.Thread(target=runner, daemon=True)
        self._thread.start()
        while self._loop is None or self._queue is None:
            time.sleep(0.01)

    async def _worker(self) -> None:
        if not self._queue:
            return
        http_client: Optional[httpx.AsyncClient] = None
        webhook = None
        try:
            if self.url or self.discord_url:
                http_client = httpx.AsyncClient(timeout=10.0)
            if self.discord_url and DiscordWebhook is not None:
                webhook = DiscordWebhook(self.discord_url)
            while True:
                payload = await self._queue.get()
                if payload is None:
                    break
                if http_client and self.url:
                    try:
                        await http_client.post(self.url, json=payload)
                    except Exception:
                        pass
                if self.discord_url:
                    content = format_discord_message(payload)
                    if content:
                        if webhook:
                            try:
                                await webhook.send_message(content=content)
                            except Exception:
                                pass
                        elif http_client:
                            try:
                                await http_client.post(self.discord_url, json={"content": content})
                            except Exception:
                                pass
        finally:
            if webhook:
                try:
                    await webhook.close()
                except Exception:
                    pass
            if http_client:
                await http_client.aclose()
        if self._loop:
            self._loop.stop()

    def send(self, event: str, message: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        if not (self.url or self.discord_url) or not self._loop or not self._queue:
            return
        payload = dict(self.base_payload)
        payload["event"] = event
        payload["timestamp"] = time.time()
        if message is not None:
            payload["message"] = message
        if extra:
            payload.update(extra)
        self._loop.call_soon_threadsafe(self._queue.put_nowait, payload)

    def close(self) -> None:
        if not (self.url or self.discord_url) or not self._loop or not self._queue or not self._thread:
            return
        self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
        self._thread.join(timeout=5)


class RunpodGraphQLClient:
    def __init__(self, api_key: str, timeout: float = 30.0) -> None:
        self.api_key = api_key
        self.client = httpx.Client(timeout=timeout)

    def query(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{RUNPOD_GRAPHQL_URL}?api_key={self.api_key}"
        response = self.client.post(url, json={"query": query, "variables": variables or {}})
        response.raise_for_status()
        payload = response.json()
        if "errors" in payload:
            message = payload["errors"][0].get("message", "Unknown GraphQL error")
            raise RuntimeError(message)
        return payload["data"]

    def get_gpu_types(self) -> List[Dict[str, Any]]:
        query = "query GpuTypes { gpuTypes { id displayName memoryInGb } }"
        data = self.query(query)
        return data.get("gpuTypes", [])

    def create_pod(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        query = """
        mutation PodFindAndDeploy($input: PodFindAndDeployOnDemandInput!) {
          podFindAndDeployOnDemand(input: $input) {
            id
            name
            imageName
            machineId
            machine { podHostId }
          }
        }
        """
        data = self.query(query, {"input": input_data})
        return data["podFindAndDeployOnDemand"]

    def get_pod_ports(self, pod_id: str) -> List[Dict[str, Any]]:
        query = """
        query Pod($podId: String!) {
          pod(input: { podId: $podId }) {
            id
            runtime {
              ports { ip isIpPublic privatePort publicPort type }
            }
          }
        }
        """
        data = self.query(query, {"podId": pod_id})
        pod = data.get("pod") or {}
        runtime = pod.get("runtime") or {}
        return runtime.get("ports") or []

    def stop_pod(self, pod_id: str) -> Dict[str, Any]:
        query = """
        mutation PodStop($podId: String!) {
          podStop(input: { podId: $podId }) {
            id
            desiredStatus
          }
        }
        """
        data = self.query(query, {"podId": pod_id})
        return data["podStop"]

    def terminate_pod(self, pod_id: str) -> Dict[str, Any]:
        query = """
        mutation PodTerminate($podId: String!) {
          podTerminate(input: { podId: $podId }) {
            id
            desiredStatus
          }
        }
        """
        data = self.query(query, {"podId": pod_id})
        return data["podTerminate"]

    def get_pod_status(self, pod_id: str) -> Dict[str, Any]:
        query = """
        query Pod($podId: String!) {
          pod(input: { podId: $podId }) {
            id
            desiredStatus
            locked
          }
        }
        """
        data = self.query(query, {"podId": pod_id})
        return data.get("pod") or {}

    def delete_pod(self, pod_id: str) -> None:
        url = f"{RUNPOD_REST_URL}/pods/{pod_id}"
        response = self.client.delete(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        if response.status_code in (200, 202, 204, 404):
            return
        raise RuntimeError(
            f"REST pod delete failed ({response.status_code}): {response.text}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Provision a Runpod Pod and run Open R1 GRPO or SFT training."
    )
    parser.add_argument("--repo-id", default="unsloth/gemma-2b", help="Hugging Face model repo id.")
    parser.add_argument("--gpu-count", type=int, default=1, help="GPU count (1, 2, 4, 6, 8).")
    parser.add_argument("--gpu-type", default=None, help="GPU type id or name substring to target.")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Docker image for the Pod.")
    parser.add_argument("--use-thinnka-image", action="store_true", help="Use reeeon/thinnka:latest.")
    parser.add_argument("--pod-name", default=None, help="Custom Pod name.")
    parser.add_argument("--cloud-type", default="ALL", choices=["ALL", "SECURE", "COMMUNITY"])
    parser.add_argument("--volume-gb", type=int, default=300, help="Persistent volume size in GB.")
    parser.add_argument("--container-disk-gb", type=int, default=None, help="Container disk size in GB.")
    parser.add_argument("--min-vcpu", type=int, default=4)
    parser.add_argument("--min-memory", type=int, default=16)
    parser.add_argument("--dataset-name", default="open-r1/OpenR1-Math-220k")
    parser.add_argument("--dataset-prompt-column", default="problem")
    parser.add_argument("--reasoning-tag", default="<think>")
    parser.add_argument("--reasoning-end-tag", default="</think>")
    parser.add_argument("--answer-tag", default="")
    parser.add_argument("--answer-end-tag", default="")
    parser.add_argument(
        "--chat-template",
        default=None,
        help="Chat template string or path to a Jinja template file.",
    )
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--per-device-train-batch", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=16)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument("--sft", action="store_true", help="Use Open R1 SFT training instead of GRPO.")
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention implementation used by transformers.",
    )
    parser.add_argument(
        "--deepspeed-stage",
        type=int,
        default=3,
        choices=[2, 3],
        help="DeepSpeed ZeRO stage for Accelerate config.",
    )
    parser.add_argument(
        "--use-vllm",
        dest="use_vllm",
        action="store_true",
        help="Use vLLM for generation (faster, higher memory use).",
    )
    parser.add_argument(
        "--no-vllm",
        "--disable-vllm",
        dest="use_vllm",
        action="store_false",
        help="Disable vLLM and use transformers generation (slower, safer).",
    )
    parser.set_defaults(use_vllm=True)
    parser.add_argument("--vllm-mode", default="colocate")
    parser.add_argument("--report-to", default="none", help="Set to wandb to enable W&B logging.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-setup", action="store_true", help="Skip Open R1 setup (useful with custom image).")
    parser.add_argument("--dry-run", action="store_true", help="Validate and select GPU only.")
    parser.add_argument("--ssh-private-key", default=None, help="Path to SSH private key.")
    parser.add_argument("--ssh-public-key", default=None, help="SSH public key string.")
    parser.add_argument("--progress-url", default=None, help="Override PROGRESS_WEBHOOK_URL.")
    parser.add_argument(
        "--discord-webhook-url",
        "--discord-webhook",
        dest="discord_webhook_url",
        default=None,
        help="Discord webhook URL for progress logs.",
    )
    parser.add_argument("--ssh-timeout-min", type=int, default=20)
    parser.add_argument("--env-file", default=None, help="Path to a .env file (defaults to .env).")
    parser.add_argument("--debug-remote", action="store_true", help="Enable verbose remote shell output.")
    return parser.parse_args()


def load_env_file(env_file: Optional[str]) -> None:
    if env_file:
        load_dotenv(env_file, override=True)
        return
    env_path = find_dotenv(filename=".env", usecwd=True)
    if env_path:
        load_dotenv(env_path, override=True)


def cleanup_pod(
    client: Optional[RunpodGraphQLClient],
    pod_id: Optional[str],
    reporter: ProgressReporter,
) -> None:
    if not client or not pod_id:
        return
    reporter.send("cleanup_start", f"Stopping and terminating pod {pod_id}.")
    try:
        client.stop_pod(pod_id)
        reporter.send("pod_stopped", f"Pod {pod_id} stopped.")
    except Exception as exc:
        reporter.send("cleanup_error", f"Failed to stop pod {pod_id}: {exc}")

    for attempt in range(12):
        try:
            status = client.get_pod_status(pod_id)
        except Exception as exc:
            reporter.send("cleanup_error", f"Failed to fetch pod status for {pod_id}: {exc}")
            status = {}
        desired = str(status.get("desiredStatus") or "").upper()
        locked = bool(status.get("locked")) if status else False
        if locked:
            reporter.send("cleanup_wait", f"Pod {pod_id} is locked; waiting before termination.")
        if desired in {"RUNNING", "RESTARTING"}:
            reporter.send("cleanup_wait", f"Pod {pod_id} is {desired}; waiting to terminate.")
            time.sleep(10)
            continue
        if desired:
            reporter.send("cleanup_wait", f"Pod {pod_id} status {desired}; attempting termination.")
        break

    for attempt in range(1, 6):
        try:
            client.delete_pod(pod_id)
            reporter.send("pod_terminated", f"Pod {pod_id} terminated.")
            return
        except Exception as exc:
            reporter.send(
                "cleanup_error",
                f"Failed to terminate pod {pod_id} (attempt {attempt}/5): {exc}",
            )
            time.sleep(10)


def ensure_model_not_gated(api: HfApi, repo_id: str, token: str):
    try:
        info = api.model_info(repo_id, token=token, files_metadata=True)
    except GatedRepoError as exc:
        raise RuntimeError(f"Model is gated: {repo_id}") from exc
    except RepositoryNotFoundError as exc:
        raise RuntimeError(f"Model not found: {repo_id}") from exc
    except HfHubHTTPError as exc:
        raise RuntimeError(f"Hugging Face error: {exc}") from exc
    if getattr(info, "gated", False):
        raise RuntimeError(f"Model is gated: {repo_id}")
    return info


def compute_model_size_gb(model_info) -> float:
    total_bytes = 0
    weight_bytes = 0
    for sibling in model_info.siblings or []:
        size = getattr(sibling, "size", None)
        if size is None:
            continue
        total_bytes += size
        filename = getattr(sibling, "rfilename", "") or ""
        if any(filename.endswith(ext) for ext in WEIGHT_EXTENSIONS):
            weight_bytes += size
    if weight_bytes == 0:
        weight_bytes = total_bytes
    return weight_bytes / (1024 ** 3)


def resolve_chat_template(
    api: HfApi,
    repo_id: str,
    token: str,
    model_info,
    template_arg: Optional[str],
) -> Optional[str]:
    if template_arg:
        candidate = Path(template_arg)
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
        return template_arg

    siblings = [s.rfilename for s in (model_info.siblings or [])]
    if "chat_template.jinja" in siblings:
        return None

    if "tokenizer_config.json" in siblings:
        try:
            config_path = hf_hub_download(repo_id, "tokenizer_config.json", token=token)
            data = json.loads(Path(config_path).read_text(encoding="utf-8"))
            if data.get("chat_template"):
                return None
        except Exception:
            pass

    return DEFAULT_CHAT_TEMPLATE


def select_gpu_candidates(
    gpu_types: List[Dict[str, Any]],
    required_vram_gb: float,
    explicit_choice: Optional[str] = None,
    gpu_filter: Optional[str] = None,
) -> Iterable[Tuple[str, Dict[str, Any]]]:
    normalized_filter = (gpu_filter or "").strip().lower()
    filter_is_set = bool(normalized_filter)
    for label, matcher in GPU_PRIORITY:
        if explicit_choice and explicit_choice != label:
            continue
        matches = []
        for gpu in gpu_types:
            name = f"{gpu.get('displayName', '')} {gpu.get('id', '')}".lower()
            if filter_is_set and normalized_filter not in name:
                continue
            if matcher(name):
                memory = gpu.get("memoryInGb") or 0
                if memory >= required_vram_gb:
                    matches.append(gpu)
        matches.sort(key=lambda g: g.get("memoryInGb", 0), reverse=True)
        for match in matches:
            yield label, match


def build_pod_env(ssh_public_key: str, hf_token: str) -> List[Dict[str, str]]:
    env = [
        {"key": "HF_TOKEN", "value": hf_token},
        {"key": "HUGGINGFACE_HUB_TOKEN", "value": hf_token},
        {"key": "HF_HOME", "value": "/workspace/.cache/huggingface"},
    ]
    if ssh_public_key:
        env.append({"key": "SSH_PUBLIC_KEY", "value": ssh_public_key})
        env.append({"key": "PUBLIC_KEY", "value": ssh_public_key})
    return env


def ensure_ssh_key_material(args: argparse.Namespace) -> Tuple[str, Path]:
    if args.ssh_public_key:
        public_key = args.ssh_public_key.strip()
    else:
        public_key = os.getenv("SSH_PUBLIC_KEY") or os.getenv("RUNPOD_SSH_PUBLIC_KEY") or ""
        if not public_key:
            default_pub = Path.home() / ".ssh" / "id_ed25519.pub"
            if default_pub.exists():
                public_key = default_pub.read_text(encoding="utf-8").strip()
    if not public_key:
        raise RuntimeError("SSH public key not found. Set SSH_PUBLIC_KEY or --ssh-public-key.")

    private_key_path = args.ssh_private_key or os.getenv("SSH_PRIVATE_KEY_PATH")
    if private_key_path:
        private_key_file = Path(private_key_path).expanduser()
    else:
        private_key_file = Path.home() / ".ssh" / "id_ed25519"
    if not private_key_file.exists():
        raise RuntimeError(f"SSH private key not found: {private_key_file}")
    return public_key, private_key_file


def load_private_key(path: Path) -> paramiko.PKey:
    try:
        return paramiko.Ed25519Key.from_private_key_file(str(path))
    except paramiko.SSHException:
        return paramiko.RSAKey.from_private_key_file(str(path))


def wait_for_ssh_port(
    client: RunpodGraphQLClient, pod_id: str, timeout_min: int
) -> Tuple[str, int]:
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        ports = client.get_pod_ports(pod_id)
        for port in ports:
            if int(port.get("privatePort") or 0) == 22:
                ip = port.get("ip")
                public_port = port.get("publicPort")
                if ip and public_port:
                    return ip, int(public_port)
        time.sleep(10)
    raise TimeoutError("SSH port did not become available in time.")


def build_grpo_config(
    args: argparse.Namespace,
    hub_model_id: str,
    chat_template: Optional[str],
) -> Dict[str, Any]:
    default_reasoning = args.reasoning_tag == "<think>" and args.reasoning_end_tag == "</think>"
    default_answer = args.answer_tag == "<answer>" and args.answer_end_tag == "</answer>"
    use_default_tags = default_reasoning and default_answer
    if use_default_tags:
        reward_funcs = ["accuracy", "format", "tag_count"]
    else:
        reward_funcs = ["accuracy"]

    report_to = [] if args.report_to == "none" else [args.report_to]
    output_dir = args.output_dir or f"/workspace/models/{hub_model_id.replace('/', '-')}"

    if args.reasoning_tag and args.reasoning_end_tag:
        reasoning_instruction = (
            f"First reason inside {args.reasoning_tag} and {args.reasoning_end_tag}."
        )
    else:
        reasoning_instruction = "First think step by step."

    if args.answer_tag and args.answer_end_tag:
        answer_instruction = (
            f"Then provide the final answer inside {args.answer_tag} and {args.answer_end_tag}."
        )
    else:
        answer_instruction = "Then provide the final answer directly after thinking."

    system_prompt = (
        "You are a helpful AI Assistant that provides well-reasoned and detailed responses. "
        f"{reasoning_instruction} {answer_instruction}"
    )

    effective_batch = args.per_device_train_batch * args.gradient_accumulation_steps * args.gpu_count
    requested_generations = max(1, args.num_generations)
    divisors = [d for d in range(1, requested_generations + 1) if effective_batch % d == 0]
    if divisors:
        num_generations = max(divisors)
    else:
        num_generations = 1

    config: Dict[str, Any] = {
        "model_name_or_path": args.repo_id,
        "model_revision": "main",
        "torch_dtype": "bfloat16",
        "attn_implementation": args.attn_implementation,
        "dataset_name": args.dataset_name,
        "dataset_prompt_column": args.dataset_prompt_column,
        "system_prompt": system_prompt,
        "bf16": True,
        "use_vllm": args.use_vllm,
        "do_eval": False,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "hub_model_id": hub_model_id,
        "hub_strategy": "every_save",
        "learning_rate": 1.0e-06,
        "log_completions": True,
        "log_level": "info",
        "logging_first_step": True,
        "logging_steps": 1,
        "logging_strategy": "steps",
        "lr_scheduler_type": "cosine_with_min_lr",
        "lr_scheduler_kwargs": {"min_lr_rate": 0.1},
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "num_generations": num_generations,
        "num_train_epochs": args.num_train_epochs,
        "output_dir": output_dir,
        "overwrite_output_dir": True,
        "per_device_train_batch_size": args.per_device_train_batch,
        "push_to_hub": True,
        "report_to": report_to,
        "reward_funcs": reward_funcs,
        "reward_weights": [1.0] * len(reward_funcs),
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "seed": 42,
        "temperature": 0.7,
        "use_liger_kernel": True,
        "warmup_ratio": 0.1,
    }
    if chat_template:
        config["chat_template"] = chat_template
    return config


def apply_qlora_config(config: Dict[str, Any]) -> None:
    config.update(
        {
            "use_peft": True,
            "load_in_4bit": True,
            "use_bnb_nested_quant": True,
            "bnb_4bit_quant_type": "nf4",
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_target_modules": "all-linear",
        }
    )


def build_sft_config(
    args: argparse.Namespace,
    hub_model_id: str,
    chat_template: Optional[str],
) -> Dict[str, Any]:
    report_to = [] if args.report_to == "none" else [args.report_to]
    output_dir = args.output_dir or f"/workspace/models/{hub_model_id.replace('/', '-')}"
    max_seq_length = max(1, args.max_prompt_length + args.max_completion_length)

    config: Dict[str, Any] = {
        "model_name_or_path": args.repo_id,
        "model_revision": "main",
        "torch_dtype": "bfloat16",
        "attn_implementation": args.attn_implementation,
        "dataset_name": args.dataset_name,
        "bf16": True,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "hub_model_id": hub_model_id,
        "hub_strategy": "every_save",
        "log_level": "info",
        "logging_first_step": True,
        "logging_steps": 1,
        "logging_strategy": "steps",
        "max_seq_length": max_seq_length,
        "num_train_epochs": args.num_train_epochs,
        "output_dir": output_dir,
        "overwrite_output_dir": True,
        "per_device_train_batch_size": args.per_device_train_batch,
        "push_to_hub": True,
        "report_to": report_to,
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "seed": 42,
        "use_liger_kernel": True,
    }
    if chat_template:
        config["chat_template"] = chat_template
    apply_qlora_config(config)
    return config


def build_accelerate_config(num_processes: int, zero_stage: int) -> str:
    config = {
        "compute_environment": "LOCAL_MACHINE",
        "debug": False,
        "deepspeed_config": {
            "deepspeed_multinode_launcher": "standard",
            "offload_optimizer_device": "none",
            "offload_param_device": "none",
            "zero3_init_flag": zero_stage == 3,
            "zero3_save_16bit_model": True if zero_stage == 3 else False,
            "zero_stage": zero_stage,
        },
        "distributed_type": "DEEPSPEED",
        "downcast_bf16": "no",
        "machine_rank": 0,
        "main_training_function": "main",
        "mixed_precision": "bf16",
        "num_machines": 1,
        "num_processes": num_processes,
        "rdzv_backend": "static",
        "same_network": True,
        "tpu_env": [],
        "tpu_use_cluster": False,
        "tpu_use_sudo": False,
        "use_cpu": False,
    }
    return yaml.safe_dump(config, sort_keys=False)


def upload_text(ssh: paramiko.SSHClient, remote_path: str, content: str) -> None:
    sftp = ssh.open_sftp()
    with sftp.file(remote_path, "w") as handle:
        handle.write(content)
    sftp.close()


def run_ssh_command(
    ssh: paramiko.SSHClient,
    command: str,
    reporter: ProgressReporter,
    stage: str,
) -> None:
    transport = ssh.get_transport()
    if not transport:
        raise RuntimeError("SSH transport not available.")
    channel = transport.open_session()
    channel.get_pty()
    channel.exec_command(command)

    buffer = ""
    recent_lines = deque(maxlen=200)
    last_report = 0.0

    def handle_line(line: str) -> None:
        nonlocal last_report
        recent_lines.append(line)
        print(f"[{stage}] {line}", flush=True)
        now = time.time()
        if now - last_report >= 1.0:
            reporter.send("training_progress", line, extra={"stage": stage})
            last_report = now

    while True:
        if channel.recv_ready():
            data = channel.recv(4096).decode("utf-8", errors="ignore")
            buffer += data
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if line.strip():
                    handle_line(line.strip())
        if channel.recv_stderr_ready():
            data = channel.recv_stderr(4096).decode("utf-8", errors="ignore")
            buffer += data
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if line.strip():
                    handle_line(line.strip())
        if channel.exit_status_ready():
            break
        time.sleep(0.2)

    if buffer.strip():
        handle_line(buffer.strip())

    exit_status = channel.recv_exit_status()
    if exit_status != 0:
        tail = "\n".join(recent_lines)
        reporter.send(
            "remote_error",
            f"Remote command failed with exit status {exit_status}",
            extra={"stage": stage, "tail": tail},
        )
        if tail:
            raise RuntimeError(
                "Remote command failed with exit status "
                f"{exit_status}. Last output:\n{tail}"
            )
        raise RuntimeError(f"Remote command failed with exit status {exit_status}")


def build_setup_script(debug: bool, install_flash_attn: bool) -> str:
    debug_line = "set -x" if debug else ""
    flash_attn_lines = []
    if install_flash_attn:
        flash_attn_lines = [
            "python -m pip install flash-attn --no-build-isolation",
        ]
    else:
        flash_attn_lines = [
            "python -m pip uninstall -y flash-attn || true",
            "echo 'Skipping flash-attn install (using sdpa/eager).'",
        ]
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            debug_line,
            "export DEBIAN_FRONTEND=noninteractive",
            "",
            "clone_open_r1() {",
            "  local tries=3",
            "  local i=1",
            "  while [ $i -le $tries ]; do",
            "    rm -rf /opt/open-r1",
            "    if GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/huggingface/open-r1 /opt/open-r1; then",
            "      return 0",
            "    fi",
            "    echo \"git clone failed with status $?\"",
            "    i=$((i+1))",
            "    sleep 5",
            "  done",
            "  echo \"Falling back to tarball download\"",
            "  rm -rf /opt/open-r1 /opt/open-r1-main",
            "  mkdir -p /opt",
            "  curl -L --retry 3 --retry-delay 3 https://github.com/huggingface/open-r1/archive/refs/heads/main.tar.gz | tar -xz -C /opt",
            "  mv /opt/open-r1-main /opt/open-r1",
            "}",
            "",
            "check_venv() {",
            "  if [ ! -d /opt/openr1-venv ]; then return 1; fi",
            "  source /opt/openr1-venv/bin/activate",
            "  set +e",
            "  python - <<'PY'",
            "import sys",
            "ok = sys.version_info[:2] == (3, 11)",
            "try:",
            "    import torch",
            "    ok = ok and torch.__version__.startswith('2.6.')",
            "except Exception:",
            "    ok = False",
            "sys.exit(0 if ok else 1)",
            "PY",
            "  local status=$?",
            "  set -e",
            "  deactivate >/dev/null 2>&1 || true",
            "  return $status",
            "}",
            "",
            "apt-get update",
            "apt-get install -y git git-lfs curl tar build-essential ca-certificates ninja-build python3-venv python3-dev",
            "apt-get install -y python3.11 python3.11-venv python3.11-dev || true",
            "if command -v python3.11 >/dev/null 2>&1; then",
            "  PYTHON_BIN=python3.11",
            "else",
            "  PYTHON_BIN=python3",
            "fi",
            "command -v git >/dev/null 2>&1 || { echo 'git missing after install'; exit 1; }",
            "command -v git-lfs >/dev/null 2>&1 || { echo 'git-lfs missing after install'; exit 1; }",
            "git lfs install",
            "",
            "if [ ! -d /opt/open-r1 ]; then",
            "  clone_open_r1",
            "else",
            "  if [ ! -d /opt/open-r1/.git ]; then",
            "    clone_open_r1",
            "  else",
            "    git -C /opt/open-r1 rev-parse --is-inside-work-tree >/dev/null 2>&1 || clone_open_r1",
            "  fi",
            "fi",
            "",
            "if check_venv; then",
            "  echo 'Using existing /opt/openr1-venv'",
            "else",
            "  rm -rf /opt/openr1-venv",
            "  $PYTHON_BIN -m venv /opt/openr1-venv",
            "fi",
            "",
            "source /opt/openr1-venv/bin/activate",
            "cd /opt/open-r1",
            "python -m pip install --upgrade pip setuptools wheel",
            "python -m pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124",
            "python -m pip install vllm==0.8.5.post1",
            *flash_attn_lines,
            "GIT_LFS_SKIP_SMUDGE=1 python -m pip install -e \".[dev]\"",
            "python -m pip install peft bitsandbytes",
            "python - <<'PY'",
            "import importlib",
            "missing = []",
            "for name in ('peft', 'bitsandbytes'):",
            "    try:",
            "        importlib.import_module(name)",
            "    except Exception:",
            "        missing.append(name)",
            "if missing:",
            "    raise SystemExit(f'QLoRA deps missing: {\", \".join(missing)}')",
            "print('QLoRA deps OK')",
            "PY",
            "",
        ]
    ).strip() + "\n"


def build_train_script(
    accel_config_path: str,
    train_config_path: str,
    training_script: str,
    vllm_mode: str,
    use_vllm: bool,
    debug: bool,
) -> str:
    debug_line = "set -x" if debug else ""
    vllm_flag = f" --vllm_mode {vllm_mode}" if use_vllm else ""
    is_grpo = training_script.endswith("grpo.py")
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            debug_line,
            "TOKEN_FILE=/workspace/thinnka/hf_token",
            "set +x",
            "TOKEN=\"\"",
            "if [ -f \"$TOKEN_FILE\" ]; then TOKEN=$(cat \"$TOKEN_FILE\"); fi",
            "if [ -z \"$TOKEN\" ]; then TOKEN=${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}; fi",
            "if [ -z \"$TOKEN\" ]; then echo 'HF_TOKEN missing in pod environment.'; exit 1; fi",
            "export HF_TOKEN=\"$TOKEN\"",
            "export HUGGINGFACE_HUB_TOKEN=\"$TOKEN\"",
            "export HF_HOME=\"${HF_HOME:-/workspace/.cache/huggingface}\"",
            "mkdir -p \"$HF_HOME\"",
            "python - <<'PY'",
            "import os, pathlib, sys",
            "token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')",
            "if not token:",
            "    print('HF_TOKEN missing in pod environment.', file=sys.stderr)",
            "    sys.exit(1)",
            "hf_home = os.getenv('HF_HOME', '/workspace/.cache/huggingface')",
            "path = pathlib.Path(hf_home)",
            "path.mkdir(parents=True, exist_ok=True)",
            "(path / 'token').write_text(token.strip())",
            "PY",
            "set -x",
            "source /opt/openr1-venv/bin/activate",
            "cd /opt/open-r1",
            f"ACCEL_CONFIG=\"{accel_config_path}\"",
            "NUM_PROCS=$(awk -F': ' '/^num_processes:/{print $2}' \"$ACCEL_CONFIG\" | tr -d ' ')",
            "PHYS_GPUS=0",
            "if command -v nvidia-smi >/dev/null 2>&1; then",
            "  PHYS_GPUS=$( (nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ') || echo 0 )",
            "fi",
            "VISIBLE_GPUS=\"$PHYS_GPUS\"",
            "if [ -n \"${CUDA_VISIBLE_DEVICES:-}\" ]; then",
            "  VISIBLE_GPUS=$(python - <<'PY'",
            "import os",
            "val = os.environ.get('CUDA_VISIBLE_DEVICES', '')",
            "items = [v for v in val.split(',') if v.strip() != '']",
            "print(len(items))",
            "PY",
            "  )",
            "fi",
            "if [ \"$VISIBLE_GPUS\" -eq 0 ] && [ \"$PHYS_GPUS\" -gt 0 ]; then",
            "  export PHYS_GPUS",
            "  export CUDA_VISIBLE_DEVICES=$(python - <<'PY'",
            "import os",
            "n = int(os.environ.get('PHYS_GPUS', '0'))",
            "print(','.join(str(i) for i in range(n)))",
            "PY",
            "  )",
            "  VISIBLE_GPUS=\"$PHYS_GPUS\"",
            "fi",
            "if [ \"$VISIBLE_GPUS\" -eq 0 ]; then",
            "  echo 'No GPUs detected in the container. Check GPU allocation or CUDA_VISIBLE_DEVICES.'",
            "  exit 1",
            "fi",
            "if [ -n \"$NUM_PROCS\" ] && [ \"$VISIBLE_GPUS\" -lt \"$NUM_PROCS\" ]; then",
            "  echo \"Adjusting Accelerate num_processes from $NUM_PROCS to $VISIBLE_GPUS to match visible GPUs.\"",
            "  sed -i \"s/^num_processes: .*/num_processes: $VISIBLE_GPUS/\" \"$ACCEL_CONFIG\"",
            "fi",
            "export VISIBLE_GPUS",
            f"TRAIN_CONFIG=\"{train_config_path}\"",
            "export TRAIN_CONFIG",
            *(
                [
                    "python - <<'PY'",
                    "import os, re",
                    "visible = int(os.environ.get('VISIBLE_GPUS', '0'))",
                    "path = os.environ.get('TRAIN_CONFIG')",
                    "if path and visible > 0:",
                    "    with open(path, 'r', encoding='utf-8') as handle:",
                    "        text = handle.read()",
                    "    def get_int(key: str):",
                    "        pattern = r'^' + re.escape(key) + r':\\s*(\\d+)\\s*$'",
                    "        match = re.search(pattern, text, flags=re.M)",
                    "        return int(match.group(1)) if match else None",
                    "    per_device = get_int('per_device_train_batch_size')",
                    "    grad_accum = get_int('gradient_accumulation_steps')",
                    "    num_generations = get_int('num_generations')",
                    "    if None not in (per_device, grad_accum, num_generations):",
                    "        effective = per_device * grad_accum * visible",
                    "        if effective > 0 and effective % num_generations != 0:",
                    "            divisors = [d for d in range(1, num_generations + 1) if effective % d == 0]",
                    "            new_gen = max(divisors) if divisors else 1",
                    "            if new_gen != num_generations:",
                    "                text = re.sub(",
                    "                    r'^num_generations:\\s*\\d+\\s*$',",
                    "                    f'num_generations: {new_gen}',",
                    "                    text,",
                    "                    flags=re.M,",
                    "                )",
                    "                with open(path, 'w', encoding='utf-8') as handle:",
                    "                    handle.write(text)",
                    "                print(",
                    "                    f'Adjusted num_generations from {num_generations} to {new_gen} '",
                    "                    f'to divide effective batch {effective}.'",
                    "                )",
                    "PY",
                ]
                if is_grpo
                else []
            ),
            "ACCELERATE_LOG_LEVEL=info \\",
            f"accelerate launch --config_file {accel_config_path} \\",
            f"  {training_script} --config {train_config_path}{vllm_flag}",
            "",
        ]
    ).strip() + "\n"


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)
    if args.use_thinnka_image:
        args.image = CUSTOM_IMAGE

    if bool(args.answer_tag) ^ bool(args.answer_end_tag):
        raise RuntimeError("Answer tags must include both start and end or be empty.")
    if bool(args.reasoning_tag) ^ bool(args.reasoning_end_tag):
        raise RuntimeError("Reasoning tags must include both start and end or be empty.")

    if args.gpu_count not in ALLOWED_GPU_COUNTS:
        raise RuntimeError("GPU count must be one of 1, 2, 4, 6, 8.")

    runpod_api_key = os.getenv("RUNPOD_API_KEY")
    if not runpod_api_key:
        raise RuntimeError("RUNPOD_API_KEY is required.")

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN or HUGGINGFACE_HUB_TOKEN is required.")

    ssh_public_key, ssh_private_key_path = ensure_ssh_key_material(args)
    progress_url = args.progress_url or os.getenv("PROGRESS_WEBHOOK_URL")
    discord_webhook_url = args.discord_webhook_url or os.getenv("DISCORD_WEBHOOK_URL")

    reporter = ProgressReporter(
        progress_url,
        {"project": PROJECT_NAME, "repo_id": args.repo_id},
        discord_webhook_url,
    )
    ssh_client: Optional[paramiko.SSHClient] = None
    client: Optional[RunpodGraphQLClient] = None
    pod_id: Optional[str] = None
    error_raised = False

    try:
        reporter.send("start", "Starting run.")

        hf_api = HfApi()
        model_info = ensure_model_not_gated(hf_api, args.repo_id, hf_token)
        model_size_gb = compute_model_size_gb(model_info)
        required_vram_gb = math.ceil(model_size_gb)

        reporter.send(
            "model_checked",
            f"Model size {model_size_gb:.2f} GB, required VRAM {required_vram_gb} GB.",
        )

        client = RunpodGraphQLClient(runpod_api_key)
        gpu_types = client.get_gpu_types()

        candidates = list(select_gpu_candidates(gpu_types, required_vram_gb, gpu_filter=args.gpu_type))
        if not candidates:
            if args.gpu_type:
                raise RuntimeError(
                    f"No eligible GPU matches '{args.gpu_type}' with >= {required_vram_gb} GB VRAM. "
                    "Check available GPU types in Runpod."
                )
            raise RuntimeError("No eligible GPU type meets the VRAM requirement.")

        volume_gb = args.volume_gb or max(40, int(math.ceil(model_size_gb * 2)))
        container_disk_gb = args.container_disk_gb or max(40, int(math.ceil(model_size_gb * 1.5)))

        pod_name = args.pod_name or f"thinnka-{args.repo_id.replace('/', '-')}-{args.gpu_count}x"
        env = build_pod_env(ssh_public_key, hf_token)

        pod = None
        last_error = None
        for label, gpu in candidates:
            reporter.send("gpu_try", f"Trying {label} ({gpu.get('displayName')}).")
            input_data = {
                "cloudType": args.cloud_type,
                "gpuCount": args.gpu_count,
                "volumeInGb": volume_gb,
                "containerDiskInGb": container_disk_gb,
                "minVcpuCount": args.min_vcpu,
                "minMemoryInGb": args.min_memory,
                "gpuTypeId": gpu.get("id"),
                "name": pod_name,
                "imageName": args.image,
                "dockerArgs": "",
                "ports": "22/tcp,8888/http",
                "volumeMountPath": "/workspace",
                "env": env,
            }
            if args.dry_run:
                reporter.send("dry_run", f"Selected GPU {gpu.get('displayName')}.")
                return 0
            try:
                pod = client.create_pod(input_data)
                reporter.send("pod_created", f"Pod {pod.get('id')} created with {gpu.get('displayName')}.")
                break
            except Exception as exc:
                last_error = exc
                reporter.send("gpu_fail", f"{label} failed: {exc}")

        if pod is None:
            raise RuntimeError(f"Unable to create a Pod. Last error: {last_error}")

        pod_id = pod.get("id")
        if not pod_id:
            raise RuntimeError("Pod creation did not return a pod id.")

        reporter.send("ssh_wait", "Waiting for SSH port 22.")
        ssh_host, ssh_port = wait_for_ssh_port(client, pod_id, args.ssh_timeout_min)
        reporter.send("ssh_ready", f"SSH available at {ssh_host}:{ssh_port}.")

        private_key = load_private_key(ssh_private_key_path)
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(
            hostname=ssh_host,
            port=ssh_port,
            username="root",
            pkey=private_key,
            timeout=30,
        )

        reporter.send("ssh_connected", "SSH connected.")

        hub_model_id = args.hub_model_id
        training_mode = "sft" if args.sft else "grpo"
        if not hub_model_id:
            whoami = hf_api.whoami(token=hf_token)
            user = whoami.get("name") or whoami.get("user") or "unknown"
            base_name = args.repo_id.split("/")[-1]
            hub_model_id = f"{user}/{base_name}-{training_mode}-thinnka"

        chat_template = resolve_chat_template(hf_api, args.repo_id, hf_token, model_info, args.chat_template)
        if args.sft:
            config = build_sft_config(args, hub_model_id, chat_template)
        else:
            config = build_grpo_config(args, hub_model_id, chat_template)
        if chat_template:
            reporter.send("chat_template", "Using custom chat template.")
        if not args.sft and config.get("num_generations") != args.num_generations:
            reporter.send(
                "num_generations_adjusted",
                f"Adjusted num_generations to {config.get('num_generations')} "
                f"to match effective batch size.",
            )
        config_text = yaml.safe_dump(config, sort_keys=False)
        deepspeed_stage = args.deepspeed_stage
        if args.sft and deepspeed_stage == 3:
            deepspeed_stage = 2
            reporter.send(
                "deepspeed_adjusted",
                "QLoRA SFT is incompatible with ZeRO-3; using ZeRO-2 instead.",
            )
        accel_config_text = build_accelerate_config(args.gpu_count, deepspeed_stage)
        remote_dir = "/workspace/thinnka"
        remote_config_path = f"{remote_dir}/{training_mode}_config.yaml"
        remote_accel_config_path = f"{remote_dir}/accelerate.yaml"
        remote_token_path = f"{remote_dir}/hf_token"

        run_ssh_command(ssh_client, f"bash -lc \"mkdir -p {remote_dir}\"", reporter, "setup")
        upload_text(ssh_client, remote_config_path, config_text)
        upload_text(ssh_client, remote_accel_config_path, accel_config_text)
        upload_text(ssh_client, remote_token_path, hf_token)
        run_ssh_command(ssh_client, f"bash -lc \"chmod 600 {remote_token_path}\"", reporter, "setup")
        reporter.send("config_uploaded", f"Config uploaded to {remote_config_path}.")
        reporter.send("config_uploaded", f"Accelerate config uploaded to {remote_accel_config_path}.")

        if not args.skip_setup:
            reporter.send("setup_start", "Installing Open R1 on the Pod.")
            install_flash_attn = args.attn_implementation == "flash_attention_2"
            setup_script = build_setup_script(args.debug_remote, install_flash_attn)
            remote_setup_path = f"{remote_dir}/setup.sh"
            upload_text(ssh_client, remote_setup_path, setup_script)
            run_ssh_command(ssh_client, f"bash -lc \"chmod +x {remote_setup_path}\"", reporter, "setup")
            run_ssh_command(ssh_client, f"bash -lc \"{remote_setup_path}\"", reporter, "setup")
            reporter.send("setup_done", "Open R1 setup completed.")
        else:
            reporter.send("setup_skip", "Skipping Open R1 setup.")

        reporter.send("train_start", f"Starting {training_mode.upper()} training.")
        train_script = build_train_script(
            remote_accel_config_path,
            remote_config_path,
            "src/open_r1/sft.py" if args.sft else "src/open_r1/grpo.py",
            args.vllm_mode,
            args.use_vllm if not args.sft else False,
            args.debug_remote,
        )
        remote_train_path = f"{remote_dir}/train.sh"
        upload_text(ssh_client, remote_train_path, train_script)
        run_ssh_command(ssh_client, f"bash -lc \"chmod +x {remote_train_path}\"", reporter, "train")
        run_ssh_command(ssh_client, f"bash -lc \"{remote_train_path}\"", reporter, "train")
        reporter.send("train_done", "Training finished.")
        reporter.send("done", "Run completed.")
        return 0
    except Exception as exc:
        error_raised = True
        reporter.send("error", str(exc))
        raise
    finally:
        if ssh_client:
            ssh_client.close()
        if error_raised:
            cleanup_pod(client, pod_id, reporter)
        reporter.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
