import hashlib
import json
import re
from pathlib import Path
import pandas as pd
from common import ROOT, load_config, text_sha256, write_manifest

REQUIRED_COLS = {
    "scenario_id", "lang", "role", "subcategory", "text", "origin", "reviewer", "confirmed"
}

def deterministic_split(scenario_id: str, seed: int, splits: dict) -> str:
    digest = hashlib.sha256(f"{seed}:{scenario_id}".encode("utf-8")).hexdigest()
    val = int(digest[:8], 16) / 0xFFFFFFFF
    if val < splits["calibrate"]:
        return "calibrate"
    elif val < splits["calibrate"] + splits["dev"]:
        return "dev"
    return "test"

def scrub_pii(text: str) -> str:
    # Mask real Kenyan/international mobile numbers or 10-char M-Pesa codes with synthetic placeholders
    text = re.sub(r"\b(?:\+?254|0)[17]\d{8}\b", "0700000000", text)
    text = re.sub(r"\b[A-Z0-9]{10}\b", "QWE123XYZ0", text)
    return text.strip()

def run_curation() -> None:
    cfg = load_config()
    raw_path = ROOT / "data" / "seeds" / "all_prompts_raw.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing seed file: {raw_path}")

    rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(rows)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Enforce fluent reviewer confirmation on sw and mix rows
    unreviewed = df[(df["lang"].isin(["sw", "mix"])) & (~df["confirmed"].astype(bool))]
    if len(unreviewed) > 0:
        raise ValueError(f"Found {len(unreviewed)} unreviewed sw/mix rows. Confirm all before curating.")

    df["text"] = df["text"].apply(scrub_pii)
    df["sha256"] = df["text"].apply(text_sha256)

    # Deduplicate exact texts within same language/role
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["lang", "role", "sha256"]).copy()

    # Keep only complete 6-row scenarios (2 roles x 3 languages)
    counts = df.groupby("scenario_id").size()
    valid_scenarios = counts[counts == 6].index
    df = df[df["scenario_id"].isin(valid_scenarios)].copy()

    # Assign scenario-level splits (prevents translation/twin leakage)
    df["split"] = df["scenario_id"].apply(
        lambda sid: deterministic_split(sid, cfg["seeds"]["split"], cfg["splits"])
    )

    # Hard leak assertions
    for s1 in ["calibrate", "dev", "test"]:
        for s2 in ["calibrate", "dev", "test"]:
            if s1 >= s2:
                continue
            ids1 = set(df[df["split"] == s1]["scenario_id"])
            ids2 = set(df[df["split"] == s2]["scenario_id"])
            assert ids1.isdisjoint(ids2), f"Scenario leak between {s1} and {s2}"

            h1 = set(df[df["split"] == s1]["sha256"])
            h2 = set(df[df["split"] == s2]["sha256"])
            assert h1.isdisjoint(h2), f"Text hash leak between {s1} and {s2}"

    out_dir = ROOT / "data" / "curated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "dataset.jsonl"
    df.to_json(out_jsonl, orient="records", lines=True, force_ascii=False)

    manifest_summary = {
        "total_rows": len(df),
        "total_scenarios": df["scenario_id"].nunique(),
        "dropped_duplicates": before_dedup - len(df),
        "split_counts": df["split"].value_counts().to_dict(),
        "lang_counts": df["lang"].value_counts().to_dict(),
        "dataset_sha256": hashlib.sha256(out_jsonl.read_bytes()).hexdigest(),
    }
    with open(out_dir / "curation_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_summary, f, indent=2)

    write_manifest("s01_curate", manifest_summary)
    print("[OK] Curation complete:", json.dumps(manifest_summary, indent=2))

if __name__ == "__main__":
    run_curation()
