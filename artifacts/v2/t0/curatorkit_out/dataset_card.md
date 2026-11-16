# Dataset Card: aya_safety_t0_smoke

## Summary

- **Run timestamp**: 2026-11-16T00:54:08.456813Z
- **Wall clock time**: 0.1s
- **Pipeline config hash**: `cc40ff556b9872ad`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 5 | 5 | — | 0 | 5 |
| SchemaGate | 5 | 5 | 0 | 0 | 5 |
| ExactDeduplicator | 5 | 5 | — | — | 5 |
| MinHashDeduplicator | 5 | 5 | — | — | 5 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 5/0 passed (N/A), 0 rejected
- **SchemaGate**: 5/5 passed (100.0%), 0 rejected


**Rejection reasons:**
  - None

## Deduplication

  - near_duplicates_removed: 0
  - surviving_count: 5

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
