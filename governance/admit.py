"""Confirm Node D answered through LiteLLM, then confirm the card is free.

The virtual key stays in the process environment. It is never written to the receipt.
A cloud fallback model is a refusal: the probe alias must be the one with no fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from governance.redact import public_mapping

DEFAULT_BASE = "https://ai-gateway.fimpo.net/v1"
DEFAULT_MODEL = "robo-arm-gpu-probe"
MIN_FREE_MIB = 8192
CLOUD_MARKERS = (
    "gemini",
    "gpt-",
    "gpt4",
    "claude",
    "openrouter",
    "anthropic",
    "openai",
    "moonshot",
    "kimi",
)

GpuQuery = Callable[[], dict[str, Any]]
Probe = Callable[[str, str, str], dict[str, Any]]


def _cloud(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in CLOUD_MARKERS)


def _local(name: str, requested: str) -> bool:
    lowered = name.lower()
    if _cloud(lowered):
        return False
    if "node-d" in lowered or lowered == "robo-arm-gpu-probe":
        return True
    return lowered == requested.lower() and (
        "node-d" in requested.lower() or requested.lower() == "robo-arm-gpu-probe"
    )


def parse_memory(text: str) -> int:
    line = ""
    for raw in text.splitlines():
        if raw.strip():
            line = raw.strip()
            break
    if not line:
        raise ValueError("nvidia-smi returned no memory row")
    free = int(line.split(",")[0].strip())
    return free


def parse_processes(text: str) -> list[str]:
    names: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.lower().startswith("process_name"):
            continue
        names.append(line.split(",")[0].strip())
    return names


def query_gpu(run: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Read free MiB and compute-app names via nvidia-smi."""
    import subprocess

    runner = run or subprocess.run

    def _call(args: list[str]) -> str:
        completed = runner(args, capture_output=True, text=True, check=False)
        if isinstance(completed, str):
            return completed
        code = getattr(completed, "returncode", 1)
        if code != 0:
            err = getattr(completed, "stderr", "") or ""
            raise RuntimeError(err.strip() or f"nvidia-smi exited {code}")
        return getattr(completed, "stdout", "") or ""

    memory = _call(
        ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"]
    )
    apps = _call(
        ["nvidia-smi", "--query-compute-apps=process_name,used_gpu_memory", "--format=csv,noheader"]
    )
    return {"free_mib": parse_memory(memory), "processes": parse_processes(apps)}


def probe_gateway(base_url: str, key: str, model: str, *, timeout: float = 30.0) -> dict[str, Any]:
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = url + "/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with the single word ready."}],
            "max_tokens": 8,
            "temperature": 0,
            "metadata": {
                "tenant": "robo-arm",
                "app": "robo-arm",
                "paas": "fimpo",
                "stage": "train",
            },
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            headers = {k.lower(): v for k, v in response.headers.items()}
            status = getattr(response, "status", 200)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"gateway HTTP {exc.code}: {detail}") from exc
    latency_ms = (time.perf_counter() - started) * 1000.0
    served = str(payload.get("model") or "")
    header_model = str(headers.get("x-litellm-model-id") or "")
    return {
        "status": status,
        "model": served,
        "header_model": header_model,
        "request_id": str(headers.get("x-litellm-call-id") or headers.get("x-request-id") or ""),
        "probe_latency_ms": round(latency_ms, 1),
    }


def admit(
    *,
    base_url: str | None = None,
    key: str | None = None,
    model: str | None = None,
    gpu: GpuQuery | None = None,
    probe: Probe | None = None,
    min_free_mib: int = MIN_FREE_MIB,
) -> dict[str, Any]:
    """Return a public receipt. ``admitted`` is true only when both checks pass."""
    requested = model or os.environ.get("ROBOARM_GPU_MODEL") or DEFAULT_MODEL
    gateway = base_url or os.environ.get("AI_GATEWAY_BASE_URL") or DEFAULT_BASE
    token = key if key is not None else os.environ.get("AI_GATEWAY_KEY") or ""
    receipt: dict[str, Any] = {
        "admitted": False,
        "reason": "unconfigured",
        "requested_model": requested,
        "model": "",
        "worker": "",
        "request_id": "",
        "probe_latency_ms": None,
        "gpu_free_mib": None,
        "route": "unconfigured",
    }
    if not token:
        receipt["reason"] = "gateway-key-missing"
        return public_mapping(receipt)

    do_probe = probe or (lambda base, token_, requested_: probe_gateway(base, token_, requested_))
    try:
        probed = do_probe(gateway, token, requested)
    except Exception as exc:
        receipt["reason"] = "probe-failed"
        receipt["route"] = "error"
        receipt["error"] = str(exc).replace(token, "***")
        return public_mapping(receipt)

    served = str(probed.get("model") or "")
    header_model = str(probed.get("header_model") or "")
    receipt["model"] = served
    receipt["request_id"] = str(probed.get("request_id") or "")
    receipt["probe_latency_ms"] = probed.get("probe_latency_ms")
    if _cloud(served) or (header_model and _cloud(header_model)):
        receipt["reason"] = "cloud-fallback"
        receipt["route"] = "cloud"
        return public_mapping(receipt)
    if not _local(served, requested):
        receipt["reason"] = "unexpected-model"
        receipt["route"] = "other"
        return public_mapping(receipt)
    receipt["route"] = "node-d"
    receipt["worker"] = "node-d"

    try:
        stats = (gpu or query_gpu)()
    except FileNotFoundError:
        receipt["reason"] = "nvidia-smi-missing"
        return public_mapping(receipt)
    except Exception as exc:
        receipt["reason"] = "gpu-query-failed"
        receipt["error"] = str(exc)
        return public_mapping(receipt)

    free = stats.get("free_mib")
    processes = [str(name) for name in stats.get("processes") or []]
    receipt["gpu_free_mib"] = free
    if any("vllm" in name.lower() for name in processes):
        receipt["reason"] = "vllm-holding-gpu"
        return public_mapping(receipt)
    if free is None or int(free) < min_free_mib:
        receipt["reason"] = "gpu-memory"
        return public_mapping(receipt)
    receipt["admitted"] = True
    receipt["reason"] = "ok"
    return public_mapping(receipt)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LiteLLM GPU probe plus a free-card check")
    parser.add_argument("--receipt", help="Write the public receipt JSON here")
    args = parser.parse_args(argv)
    receipt = admit()
    text = json.dumps(receipt, indent=2)
    print(text)
    if args.receipt:
        from pathlib import Path

        Path(args.receipt).write_text(text + "\n", encoding="utf-8")
    return 0 if receipt.get("admitted") else 2


if __name__ == "__main__":
    raise SystemExit(main())
