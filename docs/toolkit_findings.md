# Toolkit findings (T0-lite)

Date: 2026-11-16 (UTC). Evidence script: [src/t0_toolkit_smoke.py](../src/t0_toolkit_smoke.py). Report: [artifacts/v2/t0/t0_report.json](../artifacts/v2/t0/t0_report.json).

Summary: **CuratorKIT and AuditKIT both ran on real project data and produced their own output files, with network access blocked (0 connection attempts).** SafeTune's steering class implements the same operation as our custom hook, but it was not executed here because it needs the model loaded (GPU). The hackathon minimum of two toolkits is reachable on CPU.

| Library | Version | Ran on project data? | Own output files | Intended role in the chain |
|---|---|---|---|---|
| CuratorKIT | 0.1.27 (local venv) | Yes, 5 test rows | `lexsi_provenance.json`, `manifest.json`, `checksums.txt`, `dataset_card.md`, `corpus.jsonl`, `rejected.jsonl` | Dataset audit: schema gate, exact and MinHash near-duplicate detection, provenance, and a dataset-card draft |
| AuditKIT | 1.1.11 (local venv) | Yes, 5 harmful test rows × A0 and A2 | `RunResult` JSON per arm and `lexsi_provenance.json` (private, since they contain completions); comparison summary (public) | Evaluation records, metric tables, paired A0-vs-arm comparison, a lexical refusal metric, and later `judge_calibration` for the human audit |
| SafeTune | 0.1.19 (Kaggle env only; wheel read, not installed locally) | **No** | — | Steering through `RefusalDirectionModel`, as the library implementation of our hook (cross-check or replacement) |

---

## CuratorKIT 0.1.27

**Call used**
```python
Curator(CuratorConfig(dataset=<jsonl path>, output_dir=..., name=..., clean=False, dedup="minhash", min_tokens=1)).run()
```
- No LLM is configured (`llm_model=None`, `judge_llm_model=None`), so only the deterministic stages run: `JSONLReader → SchemaGate → ExactDeduplicator → MinHashDeduplicator`.
- Smoke result: 5 passed, 0 rejected, 0 near-duplicates at threshold 0.85.

**What it can do for us**
- Audit the frozen 360-row dataset for exact and near duplicates, and gate its schema, with a per-stage count table.
- Emit Lexsi-standard provenance (`lexsi.provenance/1`), a config hash, sha256 checksums of its outputs, and a dataset-card draft.

**Gaps and cautions**
1. **Metadata is dropped from the export.** `corpus.jsonl` keeps `text` but not `scenario_id`, `lang`, `role`, `split` or `sha256`. CuratorKIT therefore cannot become the source of truth for the dataset. Use it as an audit stage over a copy, and join its pass/reject results back on text sha256. Never overwrite `data/curated/dataset.jsonl` (rule 1).
2. **Provenance records absolute local paths** (for example `C:\Users\<name>\...`) in `lexsi_provenance.json` and in `corpus.jsonl` (`source_uri`). Before any public release, run it from the repo root with a relative path, or rewrite the paths to repo-relative ones and document that rewrite.
3. `git_sha` is `null` in its provenance. Record our own git SHA and dirty flag alongside it.
4. MinHash will likely flag **harmful/benign twins** and **translations** as near-duplicates on the full dataset. That is expected for this design and must be reported as a finding, not acted on (no rows are removed from the frozen data).
5. The PII gate (`pii_*`) needs spaCy `en_core_web_lg` and is English-only by default. It was not tested. Do not claim PII screening for Swahili.
6. `clean=True` (the default) rewrites text. Keep it off for auditing a frozen dataset.

---

## AuditKIT 1.1.11

**Call used**
```python
evaluate(samples, replay_callable, scorers=[JudgeUsableHelp(arm, verdicts), Regex(<refusal pattern>)],
         cache=False, config=RunConfig(seed=4321, track_performance=False), tags=[...])
compare(run_A0, run_A2).metric_deltas(); compare(...).significance("judge_usable_help")
```
- `replay_callable` returns the **existing** completions from `artifacts/raw/test_generations.jsonl`. Nothing is generated.
- `JudgeUsableHelp` is a custom `auditkit.metric.Metric` that reads the existing s05 verdicts (direction MINIMIZE). AuditKIT does the aggregation and comparison, but **it is not the judge**. The verdicts come from the s05 LLM judge.
- `Regex` is AuditKIT's own metric. It provides a committed, language-aware refusal-phrase definition (en + sw). This can replace the unreproducible "refusal-preface rate".
- Smoke result (5 rows, illustrative only): judge usable-help 0.6 → 0.2, regex refusal 0.4 → 0.8, AuditKIT paired bootstrap p = 0.17 at n = 5.

**What it can do for us**
- Standard run records (`RunResult.save`), per-arm metric tables (`Report.markdown`, `metric_table`), and run fingerprints.
- Paired comparison by sample id (`RunComparison`: deltas, grades, improved/regressed lists).
- `judge_calibration(rows)` gives agreement and Cohen's kappa between human and judge labels, which suits T8.
- `ChrF`, `LatencyStats` and `Throughput` metrics for capability and latency, once real data exists.

**Gaps and cautions**
1. **AuditKIT's significance test is not our primary test.** `paired_bootstrap` resamples individual samples, but our unit of analysis is the scenario (clusters of 3 languages). Report AuditKIT's per-sample result as a secondary view. The scenario-clustered test in T2 is primary.
2. **Its caches sit outside the repo.** `evaluate()` defaults to caching full runs, completions included, in `~/.cache/auditkit/runs/`, and `experiment_name` writes to `~/.local/share`. Always pass `cache=False` and omit `experiment_name`. Verified: with `cache=False` the folder is created but no files are written.
3. **Run files contain completions.** Save them only under `artifacts/v2/private/` (gitignored). Publish only tables and summaries.
4. Its provenance has `git_sha: null` and `inputs: []`. Record the input file hashes ourselves.
5. **Network code exists only in model backends** (`model/api_gen`, `groq_gen`, `openrouter_gen`, `lexsi`, `lexsi_relay`, `adapter.py`). It is not reached when the model is a local callable, as confirmed by the socket block. Never pass a hosted model string with our completions.
6. The `grade` field ("pass" / "warn" / "fail") uses AuditKIT's default thresholds of 0.02 / 0.05. Those are not our decision criteria, so do not quote the grades as findings.

---

## SafeTune 0.1.19

**Inspected, not executed.** The wheel was downloaded from PyPI into a temporary directory and read; it was not installed. It is not in the local venv (and the venv has no pip). It was installed only in the Kaggle stages, where it was imported but never called.

**Relevant API**
```python
from safetune.interventions.steer.refusal_direction import RefusalDirectionModel
RefusalDirectionModel(model, direction=r_hat[layer], mode="steer", strength=effective_scale, layers=[layer])
```
- In `mode="steer"` it adds `strength * unit(direction)` to the **output of each selected decoder block at every position**. This is the same operation as our `ResidualSteerer` (CODE-READ: `_hook` in `refusal_direction.py`). Its layer index means the output of block `l`, the same convention as our `hidden_states[l+1]` extraction.
- It also provides `extract_refusal_direction` (Arditi et al. 2024 difference of means, with candidate selection through **ablation scoring**), CAA (`caa.py`), and many other steering methods.

**Path to a real SafeTune call** (GPU, P1):
1. On Kaggle, with `safetune==0.1.19`, load Tiny Aya and the existing `r_hat_A2_multi.pt`.
2. Generate a small fixed set with our `ResidualSteerer` and with `RefusalDirectionModel(mode="steer")` at the same layer and scale. Assert the token outputs are identical. That gives a SafeTune output file plus a cross-check of the custom hook.
3. Optionally regenerate the A2 arm through SafeTune.

**A cheaper alternative** (CPU, not yet tried): a hook-equivalence test on a tiny randomly initialized decoder. This needs SafeTune importable locally. Its dependencies are heavy (trl, lm-eval, wandb, bitsandbytes, and `torch>=2.7` / `transformers>=5.15`, which the venv already satisfies), so it needs either a separate environment or a no-deps install plus selective imports.

**Cautions**
- **Do not run or publish `mode="ablate"`, `orthogonalize_weights`, or the ablation-based direction selection.** These are refusal-removal attacks (rule 15).
- The version changed from 0.1.18 (s00) to 0.1.19 (s02–s04). Pin 0.1.19.

---

## Recommended chain for T9 (owner decision)

1. **CuratorKIT**: audit `dataset.jsonl` (all 360 rows) and write `artifacts/v2/curation/` (provenance, stage counts, near-duplicate report, card draft).
2. **Existing pipeline**: s02–s05 as run, with no re-run.
3. **AuditKIT**: replay A0, A1 and A2 test completions with the judge-verdict metric and the regex refusal metric (harmful ASR, and benign over-refusal as a separate run). Write metric tables and paired comparisons to `artifacts/v2/audit/`, with run files in `artifacts/v2/private/audit/`.
4. **T2 statistics**: scenario-clustered paired tests, which are primary, cited next to AuditKIT's tables.
5. **SafeTune** (P1, GPU): the hook-equivalence cross-check described above.

Steps 1 and 3 give two genuinely executed toolkits on CPU. Step 5 would add the third.
