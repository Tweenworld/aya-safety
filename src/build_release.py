"""Assemble the Hugging Face release folders and refuse to finish unless they are provably clean.

Layout (release/ is a build product: regenerated from scratch, never edited by hand, not in git):
    release/model/    steering vectors, norms, selected configs, config, loader, model card
    release/dataset/  prompts, judge verdicts (no completions), stats, AuditKIT/CuratorKIT outputs,
                      run records, dataset card

Design decisions:
- Every input is REQUIRED. The previous version copied files "if they exist", so a missing vector
  produced a quietly incomplete release that still passed. Here a missing input is an error.
- The leak gate shares its completion fingerprints with src/prepush_check.py, so GitHub and Hugging
  Face are protected by one definition of "leak" instead of two that can drift apart.
- The gate scans every file, not a whitelist of extensions, and reads binaries as text with
  errors ignored: a completion pasted into an unexpected file type must still be caught.
- Probes (artifacts/probes/*.joblib) are deliberately not shipped: they are pooled, lexically
  separable classifiers (AGENTS.md G-REL-3) and add pickle-loading risk for no analytical value.
- capability_chrf.json is not shipped: it was computed on a 5-sentence fallback, not FLORES,
  and shipping it would invite exactly the capability claim the cards say is unsupported.
- The release manifest is written to artifacts/v2/, never artifacts/manifests/, whose files are
  frozen and hashed (AGENTS.md rule 1).

Usage: python src/build_release.py
"""
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from prepush_check import LOCAL_PATH, SECRET_PATTERNS, completion_signatures  # noqa: E402

REL = ROOT / "release"
MANIFEST = ROOT / "artifacts/v2/release_manifest.json"

# (source relative to repo root, destination relative to release/). All required.
MODEL_FILES = [
    ("artifacts/vectors/r_hat_A1_en.pt", "model/r_hat_A1_en.pt"),
    ("artifacts/vectors/r_hat_A2_multi.pt", "model/r_hat_A2_multi.pt"),
    ("artifacts/vectors/r_hat_per_lang.pt", "model/r_hat_per_lang.pt"),
    ("artifacts/vectors/mean_norms.pt", "model/mean_norms.pt"),
    ("artifacts/scores/selected_dev_configs.json", "model/selected_dev_configs.json"),
    ("configs/experiment.yaml", "model/experiment.yaml"),
    ("docs/cards/steer_example.py", "model/steer_example.py"),
    ("docs/cards/model_card.md", "model/README.md"),
]
DATASET_FILES = [
    ("data/curated/dataset.jsonl", "dataset/benchmark_prompts.jsonl"),
    ("artifacts/v2/stats/stats_tables.md", "dataset/stats/stats_tables.md"),
    ("artifacts/v2/stats/stats_summary.json", "dataset/stats/stats_summary.json"),
    ("artifacts/v2/audit/metric_tables.md", "dataset/audit/metric_tables.md"),
    ("artifacts/v2/audit/v2_audit_summary.json", "dataset/audit/v2_audit_summary.json"),
    ("artifacts/v2/curation/lexsi_provenance.json", "dataset/curation/lexsi_provenance.json"),
    ("artifacts/v2/curation/manifest.json", "dataset/curation/manifest.json"),
    ("artifacts/v2/curation/checksums.txt", "dataset/curation/checksums.txt"),
    ("artifacts/v2/curation/v2_curation_summary.json", "dataset/curation/v2_curation_summary.json"),
    ("artifacts/v2/chain_manifest.json", "dataset/run_records/chain_manifest.json"),
    ("artifacts/HASHES.txt", "dataset/run_records/HASHES.txt"),
    ("artifacts/scores/dev_sweep_summary.csv", "dataset/run_records/dev_sweep_summary.csv"),
    ("artifacts/scores/latency_benchmark.json", "dataset/run_records/latency_benchmark.json"),
    ("docs/cards/dataset_card.md", "dataset/README.md"),
]
MANIFEST_DIR = "artifacts/manifests"
JUDGE_SRC = "artifacts/scores/judge_scores.jsonl"
JUDGE_DST = "dataset/judge_verdicts.jsonl"
# Columns that would carry model output. judge_scores.jsonl has none today; this guards the future.
FORBIDDEN_COLS = {"completion", "pred_A0_base", "pred_A2_multi", "generated_text", "response", "output"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_required(pairs: list[tuple[str, str]]) -> None:
    missing = [s for s, _ in pairs if not (ROOT / s).is_file()]
    if missing:
        raise FileNotFoundError(f"Release inputs missing (refusing a partial release): {missing}")
    for src, dst in pairs:
        out = REL / dst
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / src, out)


def copy_judge_verdicts() -> None:
    rows = [json.loads(x) for x in (ROOT / JUDGE_SRC).read_text(encoding="utf-8").splitlines() if x.strip()]
    bad = FORBIDDEN_COLS & set().union(*(r.keys() for r in rows))
    if bad:
        raise RuntimeError(f"judge verdicts contain output columns {bad}; refusing to release")
    out = REL / JUDGE_DST
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def verify_release_dir(root: Path, sigs: set[str]) -> dict:
    """Raise if any file under root contains completion text, a secret or a local path.

    Kept free of global state so tests can point it at a fixture directory.
    """
    if not sigs:
        raise RuntimeError("Leak gate has zero completion signatures; an empty gate proves nothing.")
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        raise RuntimeError(f"Nothing to verify under {root}")
    problems = []
    for f in files:
        text = f.read_bytes().decode("utf-8", errors="ignore")
        flat = " ".join(text.split())
        # JSON/JSONL store newlines and quotes escaped; undo that so shingles still line up.
        unescaped = " ".join(text.replace("\\n", " ").replace('\\"', '"').split())
        if any(s in flat or s in unescaped for s in sigs):
            problems.append(f"completion text: {f}")
        if any(re.search(p, text) for p in SECRET_PATTERNS):
            problems.append(f"secret pattern: {f}")
        m = re.search(LOCAL_PATH, text)
        if m:
            problems.append(f"local path {m.group(0)[:30]!r}: {f}")
    if problems:
        raise RuntimeError("Release gate FAILED:\n  " + "\n  ".join(problems))
    return {"files_scanned": len(files), "signatures": len(sigs)}


def build() -> dict:
    if REL.exists():
        shutil.rmtree(REL)  # build product only; see module docstring
    copy_required(MODEL_FILES)
    copy_required(DATASET_FILES)
    manifests = sorted((ROOT / MANIFEST_DIR).glob("manifest_*.json"))
    if len(manifests) < 7:
        raise FileNotFoundError(f"Expected 7 stage manifests in {MANIFEST_DIR}, found {len(manifests)}")
    copy_required([(m.relative_to(ROOT).as_posix(), f"dataset/run_records/manifests/{m.name}") for m in manifests])
    copy_judge_verdicts()

    gate = verify_release_dir(REL, completion_signatures())
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    manifest = {
        "stage": "build_release_v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": sha,
        "dirty": dirty,
        "gate": gate,
        "files_sha256": {p.relative_to(REL).as_posix(): sha256(p) for p in sorted(REL.rglob("*")) if p.is_file()},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    m = build()
    print(f"[OK] release/ built and verified: {m['gate']['files_scanned']} files, "
          f"{m['gate']['signatures']} completion signatures, git {m['git_sha'][:7]}, dirty={m['dirty']}")
