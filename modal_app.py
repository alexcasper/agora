"""Run agora sweeps on Modal GPUs: one vLLM server per container, many games against it.

    modal run modal_app.py --sweep sweeps/smoke.yaml --models sweeps/models.yaml
    modal run modal_app.py --sweep sweeps/full.yaml --only qwen3-8b --tag oct26
    modal volume get agora-runs <tag> ./runs-modal      # pull results

Each model in the models file gets `shards` containers (seeds split round-robin)
on its own GPU type. Results land in the `agora-runs` volume under
<tag>/<model>/; rerunning with the same tag resumes (finished games skipped).
Weights are cached in the `agora-hf-cache` volume. Gated models need a Modal
secret holding HF_TOKEN; pass its name with AGORA_HF_SECRET.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import modal

VLLM_VERSION = os.environ.get("AGORA_VLLM_VERSION", "0.31.0")
REMOTE_ROOT = "/root/agora"

# vLLM JIT-compiles kernels at engine start and needs nvcc, so build on the CUDA devel
# image (13.4 matches the nvcc wheel vLLM 0.31.0 pulls in).
CUDA_IMAGE = os.environ.get("AGORA_CUDA_IMAGE", "nvidia/cuda:13.4.3-devel-ubuntu24.04")

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.11")
    .pip_install(f"vllm=={VLLM_VERSION}", "pyyaml>=6", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "CUDA_HOME": "/usr/local/cuda"})
    # Tool chat templates that ship in vLLM's repo but not its wheel (Phi-4-mini).
    .run_commands(
        "mkdir -p /root/templates && python -c \"import urllib.request; urllib.request.urlretrieve("
        f"'https://raw.githubusercontent.com/vllm-project/vllm/v{VLLM_VERSION}/examples/tool_chat_template_phi4_mini.jinja', "
        "'/root/templates/tool_chat_template_phi4_mini.jinja')\""
    )
    .add_local_dir(".", REMOTE_ROOT, ignore=[".venv", ".git", "runs", "**/__pycache__", "*.egg-info"])
)

hf_cache = modal.Volume.from_name("agora-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("agora-runs", create_if_missing=True)
secrets = [modal.Secret.from_name(os.environ["AGORA_HF_SECRET"])] if os.environ.get("AGORA_HF_SECRET") else []

app = modal.App("agora", image=image)


def _wait_for_server(port: int, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5):
                return
        except Exception:  # noqa: BLE001 - server still starting
            time.sleep(5)
    raise TimeoutError("vLLM did not become ready")


@app.function(
    gpu="H100",
    timeout=24 * 3600,
    volumes={"/root/.cache/huggingface": hf_cache, "/runs": runs},
    secrets=secrets,
)
def sweep_model(model: dict, sweep: dict, tag: str, shard: int = 0, shards: int = 1) -> list:
    sys.path.insert(0, REMOTE_ROOT)
    from agora.sweep import run_sweep, seed_list

    sweep = dict(sweep)
    sweep["seeds"] = seed_list(sweep.get("seeds", [0]))[shard::shards]
    port = 8000
    cmd = [
        "vllm", "serve", model["hf_id"],
        "--port", str(port),
        "--served-model-name", model["name"],
        "--max-model-len", str(model.get("max_model_len", 16384)),
        "--enable-auto-tool-choice",
        "--tool-call-parser", model.get("tool_parser", "hermes"),
        *[str(a) for a in model.get("vllm_args", [])],
    ]
    print("starting:", " ".join(cmd), flush=True)
    server = subprocess.Popen(cmd)
    stop = threading.Event()

    def commit_periodically():
        while not stop.wait(300):
            runs.commit()

    try:
        _wait_for_server(port, server, timeout=float(model.get("startup_timeout", 1800)))
        hf_cache.commit()
        runs.reload()
        threading.Thread(target=commit_periodically, daemon=True).start()
        llm = {
            "base_url": f"http://127.0.0.1:{port}/v1",
            "model": model["name"],
            "timeout": 900,
            "params": model.get("params", {"max_tokens": 2048, "temperature": 0.7}),
        }
        out = Path("/runs") / tag / model["name"]
        out.mkdir(parents=True, exist_ok=True)
        # Exact serving and sampling settings next to the results (temperature and
        # reasoning mode differ across models by design; see docs/MODELS.md).
        (out / "model.json").write_text(json.dumps({**model, "vllm_version": VLLM_VERSION, "cuda_image": CUDA_IMAGE}, indent=2))
        rows = run_sweep(sweep, llm, out, root=Path(REMOTE_ROOT), concurrency=model.get("game_concurrency"))
        return [{"model": model["name"], **r} for r in rows]
    finally:
        stop.set()
        runs.commit()
        server.terminate()


@app.local_entrypoint()
def main(sweep: str = "sweeps/smoke.yaml", models: str = "sweeps/models.yaml", only: str = "", tag: str = ""):
    import json

    import yaml

    spec = yaml.safe_load(Path(sweep).read_text())
    model_list = yaml.safe_load(Path(models).read_text())["models"]
    if only:
        wanted = set(only.split(","))
        model_list = [m for m in model_list if m["name"] in wanted]
    if not model_list:
        raise SystemExit("no models selected")
    tag = tag or f"{Path(sweep).stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    calls = []
    for m in model_list:
        fn = sweep_model.with_options(gpu=m.get("gpu", "H100"))
        shards = int(m.get("shards", 1))
        for shard in range(shards):
            calls.append((m["name"], shard, fn.spawn(m, spec, tag, shard, shards)))
    print(f"tag={tag}: spawned {len(calls)} containers for {len(model_list)} models", flush=True)
    errors = 0
    for name, shard, call in calls:
        try:
            rows = call.get()
            bad = sum(1 for r in rows if r.get("error"))
            errors += bad
            print(f"{name} shard {shard}: {len(rows)} games run this call, {bad} errors", flush=True)
        except Exception as exc:  # noqa: BLE001 - report and keep collecting other models
            errors += 1
            print(f"{name} shard {shard}: FAILED {exc}", flush=True)
    print(json.dumps({"tag": tag, "errors": errors}))
    print(f"pull results: modal volume get agora-runs {tag} ./runs-modal/{tag}")
