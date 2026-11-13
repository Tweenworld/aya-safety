import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent

def load_config() -> dict:
    cfg_path = ROOT / "configs" / "experiment.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def config_hash() -> str:
    cfg_bytes = (ROOT / "configs" / "experiment.yaml").read_bytes()
    return hashlib.sha256(cfg_bytes).hexdigest()[:16]

def text_sha256(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "uncommitted"

def write_manifest(stage: str, extra: dict | None = None) -> Path:
    out_dir = ROOT / "artifacts" / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    freeze = sorted(
        f"{dist.metadata['Name']}=={dist.version}"
        for dist in importlib.metadata.distributions()
        if dist.metadata.get("Name")
    )
    manifest = {
        "stage": stage,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "config_hash": config_hash(),
        "python_version": sys.version,
        "packages": freeze,
        "extra": extra or {},
    }
    try:
        import torch
        manifest["torch_version"] = torch.__version__
        manifest["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            manifest["gpu_name"] = torch.cuda.get_device_name(0)
            manifest["max_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    except ImportError:
        pass
    path = out_dir / f"manifest_{stage}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path
