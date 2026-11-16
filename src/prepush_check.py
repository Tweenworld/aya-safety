"""Block publication of anything that must stay private. Runs as a git pre-push hook.

Checks every file tracked at the commit being pushed (default: HEAD):
  1. Never-push paths: raw completions, audit sheet, private outputs, secrets, vectors/probes,
     the owner's local AGENTS.md.
  2. Secrets: HF / OpenAI / OpenRouter / Google API key patterns.
  3. Local machine paths (Windows user-profile paths, AppData, Linux home dirs, Kaggle working dir).
  4. Model completion text: 8-word shingles over every completion in artifacts/raw/
     (prompts are public by design and excluded from the signatures).

Usage:
    python src/prepush_check.py [<commit> ...]      # check each listed commit (default HEAD)
    python src/prepush_check.py --range <rev-range>  # check every commit in a range, e.g. main..HEAD
A push uploads every new commit, not just the tip, so the pre-push hook checks them all.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NEVER_PUSH = [
    (r"(^|/)\.env($|\.)", "secrets file"),
    (r"^artifacts/raw/", "raw completions / audit sheet"),
    (r"(^|/)private/", "private outputs (contain completions)"),
    (r"human_validation", "human audit sheet"),
    (r"\.pt$", "steering vectors (release via HF with a card, not git)"),
    (r"\.joblib$", "probe pickles (release via HF with a card, not git)"),
    (r"^AGENTS\.md$", "owner's local operating manual"),
]
SECRET_PATTERNS = [
    r"hf_[A-Za-z0-9]{30,}",
    r"sk-or-v1-[A-Za-z0-9]{20,}",
    r"sk-[A-Za-z0-9]{32,}",
    r"AIza[0-9A-Za-z_\-]{30,}",
]
LOCAL_PATH = r"(?i)([A-Z]:[\\/]+Users[\\/]+(?!<)[^\\/\s\"']+|AppData[\\/]|/home/[a-z][^/\s]*/|/kaggle/working)"
SELF = "src/prepush_check.py"
COMPLETION_SOURCES = [ROOT / "artifacts/raw/test_generations.jsonl"]
CAPABILITY_SOURCE = ROOT / "artifacts/raw/capability_generations.jsonl"


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True).stdout


def completion_signatures() -> set[str]:
    sources = [p for p in COMPLETION_SOURCES if p.exists()]
    if not sources:
        raise RuntimeError("No raw completion file found locally; cannot verify the push is leak-free.")
    prompts = " ".join(
        " ".join(json.loads(x)["text"].split())
        for x in (ROOT / "data/curated/dataset.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()
    )
    texts = []
    for p in sources:
        texts += [json.loads(x)["completion"] for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    if CAPABILITY_SOURCE.exists():
        for x in CAPABILITY_SOURCE.read_text(encoding="utf-8").splitlines():
            if x.strip():
                r = json.loads(x)
                texts += [r.get("pred_A0_base", ""), r.get("pred_A2_multi", "")]
    sigs = set()
    for t in texts:
        w = t.split()
        for i in range(0, max(len(w) - 8, 1), 4):
            chunk = " ".join(w[i:i + 8])
            if len(chunk) >= 30 and chunk not in prompts:
                sigs.add(chunk)
    if not sigs:
        raise RuntimeError("Built zero completion signatures; refusing to pass.")
    return sigs


def check_commit(commit: str, sigs: set[str]) -> list[str]:
    files = [f for f in git("ls-tree", "-r", "--name-only", commit).decode().split("\n") if f]
    problems = []
    for f in files:
        for pat, why in NEVER_PUSH:
            if re.search(pat, f):
                problems.append(f"NEVER-PUSH  {f}  ({why})")
        blob = git("show", f"{commit}:{f}").decode("utf-8", "ignore")
        # This file contains the patterns themselves; the completion check below still applies to it.
        pattern_text = "" if f == SELF else blob
        for pat in SECRET_PATTERNS:
            if re.search(pat, pattern_text):
                problems.append(f"SECRET      {f}  (pattern {pat[:12]}...)")
        m = re.search(LOCAL_PATH, pattern_text)
        if m:
            problems.append(f"LOCAL PATH  {f}  ({m.group(0)[:40]})")
        flat = " ".join(blob.split())
        unescaped = " ".join(blob.replace("\\n", " ").replace('\\"', '"').split())
        if any(s in flat or s in unescaped for s in sigs):
            problems.append(f"COMPLETION  {f}  (contains model completion text)")
    return problems


def main() -> int:
    args = sys.argv[1:]
    if args[:1] == ["--range"]:
        commits = git("rev-list", args[1]).decode().split()
    else:
        commits = args or ["HEAD"]
    sigs = completion_signatures()
    failed = 0
    for c in commits:
        short = git("rev-parse", "--short", c).decode().strip()
        problems = check_commit(c, sigs)
        if problems:
            failed += 1
            print(f"[BLOCKED] {short}: {len(problems)} problem(s)")
            print("\n".join("  " + p for p in problems))
    if failed:
        return 1
    print(f"[OK] {len(commits)} commit(s) checked: no never-push paths, secrets, local paths "
          f"or completion text ({len(sigs)} signatures).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
