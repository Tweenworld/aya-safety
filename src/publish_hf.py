"""Upload release/ to the Hugging Face Hub, privately by default.

Two-phase on purpose: publishing is the one irreversible step in this project (public files can be
mirrored or cached the moment they appear), so the default run uploads to PRIVATE repos for review
on the Hub, and only a second, explicit run with --make-public flips visibility.

Preconditions enforced before any network call:
  - release/ matches artifacts/v2/release_manifest.json byte for byte (no edits after the gate ran)
  - the leak gate passes again on the exact files about to be uploaded
  - both cards have a real license (no TODO placeholders)

Usage:
    python src/publish_hf.py --model-repo USER/name --dataset-repo USER/name            # private upload
    python src/publish_hf.py --model-repo USER/name --dataset-repo USER/name --make-public
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from build_release import REL, verify_release_dir  # noqa: E402
from prepush_check import completion_signatures  # noqa: E402

MANIFEST = ROOT / "artifacts/v2/release_manifest.json"


def check_release_unchanged() -> None:
    recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))["files_sha256"]
    actual = {p.relative_to(REL).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in REL.rglob("*") if p.is_file()}
    if recorded != actual:
        changed = sorted(set(recorded.items()) ^ set(actual.items()))
        raise RuntimeError(f"release/ differs from the gated manifest; rebuild first. Diff: {changed[:6]}")


def check_cards() -> None:
    for card in (REL / "model/README.md", REL / "dataset/README.md"):
        text = card.read_text(encoding="utf-8")
        if "TODO" in text.split("---")[1]:  # YAML front matter only; body TODOs are tracked separately
            raise RuntimeError(f"{card.relative_to(ROOT)}: license metadata still TODO; set it before publishing")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-repo", required=True)
    ap.add_argument("--dataset-repo", required=True)
    ap.add_argument("--make-public", action="store_true",
                    help="flip both repos to public (dataset stays gated); run only after reviewing on the Hub")
    args = ap.parse_args()

    check_release_unchanged()
    verify_release_dir(REL, completion_signatures())
    check_cards()

    from dotenv import load_dotenv
    from huggingface_hub import HfApi

    load_dotenv(ROOT / ".env")
    api = HfApi(token=os.environ["HF_TOKEN"])

    targets = ((args.model_repo, "model", REL / "model"), (args.dataset_repo, "dataset", REL / "dataset"))
    if args.make_public:
        for repo, kind, _ in targets:
            api.update_repo_settings(repo, repo_type=kind, private=False)
            if kind == "dataset":
                # Prompts describe real fraud patterns: require an access request with stated use.
                api.update_repo_settings(repo, repo_type=kind, gated="auto")
            print(f"[public] {kind} {repo}")
        return 0

    for repo, kind, folder in targets:
        api.create_repo(repo, repo_type=kind, private=True, exist_ok=True)
        info = api.repo_info(repo, repo_type=kind)
        if not info.private:
            # exist_ok=True does not change visibility of an existing repo; never upload into a public one
            raise RuntimeError(f"{repo} already exists and is PUBLIC; refusing to upload. Make it private first.")
        api.upload_folder(repo_id=repo, repo_type=kind, folder_path=str(folder),
                          commit_message="aya-safety v2 release (gated build, see run_records)")
        print(f"[private] uploaded {folder.relative_to(ROOT)} -> {kind} {repo}")
    print("Review both repos on the Hub, then rerun with --make-public.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
