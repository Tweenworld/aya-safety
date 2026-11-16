---
tags:
- curatorkit
configs:
- config_name: corpus
  default: true
  description: curatorkit corpus export, sha256 c8241cdcd13809e022b742de7e6a84a1e37dc1f37e43a12f8a2bc5cf75101d62
  data_files:
  - split: train
    path: corpus.jsonl
---

# Dataset Card: aya_safety_v2_dataset_audit

## Summary

- **Run timestamp**: 2026-11-16T05:50:12.651107Z
- **Wall clock time**: 2.1s
- **Pipeline config hash**: `f8b790aba0859760`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 360 | 360 | — | 0 | 360 |
| SchemaGate | 360 | 360 | 0 | 0 | 360 |
| ExactDeduplicator | 360 | 360 | — | — | 360 |
| MinHashDeduplicator | 360 | 360 | — | — | 360 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 360/0 passed (N/A), 0 rejected
- **SchemaGate**: 360/360 passed (100.0%), 0 rejected


**Rejection reasons:**
  - None

## Deduplication

  - near_duplicates_removed: 0
  - surviving_count: 360

## Known Limitations

- MinHash threshold used: **0.85** — this is a starting point, not a universal answer. Adjust per domain (legal, medical, and other domain-specific corpora may need a lower threshold to avoid losing structurally similar but semantically valid examples).
- Table extraction requires `extract_tables: true` on the PDF reader and MinerU with GPU support. Disabled by default.
- Token counting uses a whitespace tokenizer by default. For exact LLM context window
  enforcement, enable `use_tiktoken: true` in the SchemaGate config.
- Both within-run (MinHash/Diversity) and cross-run (EmbeddingDeduplicator) deduplication
  are available. Cross-run dedup persists an embedding index across runs; use
  `embedding_reset_index: true` to start fresh.

## Reproduction

```bash
pip install curatorkit
curatorkit run pipeline.yaml --output-dir ./out/
```

## Tool Versions

- **curatorkit**: 0.1.27
- **python**: 3.12.9

## Configs

Load one with `datasets.load_dataset("<this folder or Hub repo>", "<config>")`.

| Config | Splits (rows) | Columns |
|--------|---------------|---------|
| `corpus` | train (360) | `text`: string<br>`source_file, source_uri, heading, content_type, task_type, table_html`: string<br>`page, chunk_index`: int or null<br>`table_bbox`: list of number |
