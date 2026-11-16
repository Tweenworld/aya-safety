"""Record sha256 hashes of frozen artifacts so later changes are detectable.

Usage:
    python src/hash_artifacts.py            # write artifacts/HASHES.txt
    python src/hash_artifacts.py --check    # verify current files against artifacts/HASHES.txt
"""
import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "HASHES.txt"
TARGETS = [
    "data/curated/dataset.jsonl",
    "artifacts/raw",
    "artifacts/scores",
    "artifacts/vectors",
    "artifacts/probes",
    "artifacts/manifests",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect() -> dict[str, str]:
    hashes = {}
    for target in TARGETS:
        p = ROOT / target
        if not p.exists():
            raise FileNotFoundError(f"Expected artifact path is missing: {target}")
        files = [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file())
        if not files:
            raise FileNotFoundError(f"Artifact directory is empty: {target}")
        for f in files:
            hashes[f.relative_to(ROOT).as_posix()] = sha256_file(f)
    return hashes


def read_recorded() -> dict[str, str]:
    recorded = {}
    for line in OUT.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            digest, rel = line.split("  ", 1)
            recorded[rel] = digest
    return recorded


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    current = collect()

    if args.check:
        recorded = read_recorded()
        changed = sorted(k for k in recorded if k in current and current[k] != recorded[k])
        missing = sorted(k for k in recorded if k not in current)
        added = sorted(k for k in current if k not in recorded)
        for label, items in (("CHANGED", changed), ("MISSING", missing), ("NEW (not frozen)", added)):
            for k in items:
                print(f"{label}: {k}")
        if changed or missing:
            print("[FAIL] Frozen artifacts differ from artifacts/HASHES.txt")
            return 1
        print(f"[OK] {len(recorded)} frozen files match")
        return 0

    lines = [
        "# sha256 of frozen artifacts. Regenerate only for new, versioned outputs; never to hide a change.",
        f"# generated_utc: {datetime.now(timezone.utc).isoformat()}",
        f"# generator: src/hash_artifacts.py",
    ]
    lines += [f"{digest}  {rel}" for rel, digest in sorted(current.items())]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Wrote {len(current)} hashes to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
