import json
import shutil
from pathlib import Path
import pandas as pd
from common import ROOT, write_manifest

FORBIDDEN_COLS = {"completion", "pred_A0_base", "pred_A2_multi", "generated_text"}


def build_and_verify_release() -> None:
    rel_dir = ROOT / "release"
    if rel_dir.exists():
        shutil.rmtree(rel_dir)

    model_rel = rel_dir / "model_collection"
    data_rel = rel_dir / "dataset_collection"
    model_rel.mkdir(parents=True, exist_ok=True)
    data_rel.mkdir(parents=True, exist_ok=True)

    # 1. Package Model Collection (unit refusal vectors, probes, frozen config)
    for pt_file in ["r_hat_A1_en.pt", "r_hat_A2_multi.pt", "r_hat_per_lang.pt", "mean_norms.pt"]:
        src_p = ROOT / "artifacts" / "vectors" / pt_file
        if src_p.exists():
            shutil.copy2(src_p, model_rel / pt_file)

    for probe_file in (ROOT / "artifacts" / "probes").glob("*"):
        if probe_file.is_file():
            shutil.copy2(probe_file, model_rel / probe_file.name)

    shutil.copy2(ROOT / "configs" / "experiment.yaml", model_rel / "experiment.yaml")
    sel_cfg = ROOT / "artifacts" / "scores" / "selected_dev_configs.json"
    if sel_cfg.exists():
        shutil.copy2(sel_cfg, model_rel / "selected_dev_configs.json")

    # 2. Package Dataset Collection (curated prompts + labels + judge scores + CIs; ZERO raw completions)
    shutil.copy2(ROOT / "data" / "curated" / "dataset.jsonl", data_rel / "benchmark_prompts.jsonl")
    shutil.copy2(ROOT / "data" / "curated" / "curation_manifest.json", data_rel / "curation_manifest.json")

    scores_jsonl = ROOT / "artifacts" / "scores" / "judge_scores.jsonl"
    if scores_jsonl.exists():
        df_s = pd.read_json(scores_jsonl, lines=True)
        bad_cols = FORBIDDEN_COLS & set(df_s.columns)
        if bad_cols:
            raise RuntimeError(f"SECURITY VIOLATION: Forbidden columns found in judge_scores: {bad_cols}")
        df_s.to_json(data_rel / "audit_judge_scores.jsonl", orient="records", lines=True, force_ascii=False)

    for stat_file in ["final_metrics_table.csv", "capability_chrf.json", "latency_benchmark.json", "dev_sweep_summary.csv"]:
        sp = ROOT / "artifacts" / "scores" / stat_file
        if sp.exists():
            shutil.copy2(sp, data_rel / stat_file)

    # 3. Hard Leak-Check against artifacts/raw/ completions
    # Each signature is checked both verbatim and in JSON-escaped form, because
    # newlines/quotes inside a completion appear escaped (\n, \") in .jsonl/.json files.
    raw_path = ROOT / "artifacts" / "raw" / "test_generations.jsonl"
    raw_snippets = []
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                comp = json.loads(line).get("completion", "").strip()
                if len(comp) >= 40:
                    raw_snippets.append(comp[:60])

    signatures = []
    for snip in raw_snippets:
        signatures.append(snip)
        escaped = json.dumps(snip, ensure_ascii=False)[1:-1]
        if escaped != snip:
            signatures.append(escaped)

    for file_path in rel_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix in {".jsonl", ".csv", ".json", ".yaml", ".md"}:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            for sig in signatures:
                if sig in text:
                    raise RuntimeError(f"LEAK DETECTED in {file_path}: matched raw completion substring!")

    write_manifest("build_release", {"verified_clean": True, "checked_raw_snippets": len(raw_snippets)})
    print(f"[OK] Release verified clean (checked {len(raw_snippets)} raw completion signatures). Ready at {rel_dir}")


if __name__ == "__main__":
    build_and_verify_release()