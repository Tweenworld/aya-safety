import os
import time
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download

load_dotenv()

REPO = os.environ.get("STAGING_REPO", "LeoCleinte/aya-safety-staging")
TOKEN = os.environ.get("HF_TOKEN")
ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
_last_sync = 0.0

def _get_api() -> HfApi:
    if not TOKEN:
        raise RuntimeError("HF_TOKEN is not set in environment or .env")
    return HfApi(token=TOKEN)

def pull() -> None:
    api = _get_api()
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        local_dir=str(ARTIFACTS_DIR),
        token=TOKEN,
    )

def push(msg: str = "sync artifacts") -> None:
    api = _get_api()
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(ARTIFACTS_DIR),
        repo_id=REPO,
        repo_type="dataset",
        commit_message=msg,
        ignore_patterns=["*.pyc", "__pycache__/*"],
    )

def push_if_due(interval_sec: int = 300, msg: str = "periodic checkpoint") -> None:
    global _last_sync
    if time.time() - _last_sync >= interval_sec:
        push(msg)
        _last_sync = time.time()
