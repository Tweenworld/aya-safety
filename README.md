# aya-safety

Inference-time refusal steering for [Tiny Aya Earth](https://huggingface.co/CohereLabs/tiny-aya-earth) on East African mobile-money fraud requests in English, Swahili and code-mixed Swahili–English. Built for the Lexsi hackathon, Track 2.

- **Article:** see `docs/article.md` (also published on Hugging Face)
- **Steering vectors and model card:** [collection](https://huggingface.co/collections/LeoCleinte/tiny-aya-earth-swahili-and-sheng-safety-steering-6afab528fb5ebfc572c73403)
- **Benchmark, judge verdicts and dataset card (gated):** [collection](https://huggingface.co/collections/LeoCleinte/east-african-mobile-money-fraud-safety-audit-6afab6956a5b1af4d9c43752)
- **Frozen results:** git tag `results-v1-frozen`

## Result in one table

Test split: 22 scenarios × 3 languages = 66 harmful and 66 benign prompts per arm. 95% Clopper–Pearson intervals.

| Metric (of 66) | Base | + A1 English direction | + A2 multilingual direction |
|---|---|---|---|
| Usable help on harmful prompts | 35 (53.0%) [40.3, 65.4] | 22 (33.3%) [22.2, 46.0] | 21 (31.8%) [20.9, 44.4] |
| Judged refusal of benign prompts | 0 (0.0%) [0.0, 5.4] | 2 (3.0%) [0.4, 10.5] | 5 (7.6%) [2.5, 16.8] |
| Refusal phrases in benign answers | 1 (1.5%) [0.0, 8.2] | 10 (15.2%) [7.5, 26.1] | 11 (16.7%) [8.6, 27.9] |

Both arms reduce usable help compared with the base model (paired scenario-level sign-flip test, Holm p = 0.004). A1 and A2 do not differ (p = 1.0). Counting only `full_help` as usable, both drops shrink to 13.6 points (Holm p = 0.09). Full tables: `artifacts/v2/stats/stats_tables.md`.

## What this does not show

- **Direction specificity is untested.** There is no random-direction control, so a generic perturbation of the same size might have a similar effect.
- **The judge is not validated.** One free-tier LLM judge, which saw each prompt's role, produced the verdicts. There is no human audit.
- **No capability claim.** The FLORES load failed and the check fell back to 5 in-domain sentence pairs.
- **The data is not human-reviewed.** The prompts were written as literals in generator scripts. The `origin`, `reviewer` and `confirmed` fields are script constants. The Swahili and code-mixed text has not been verified by a native speaker.
- **This is not a security boundary.** Only single-turn, non-adversarial prompts were tested. Anyone with the weights can omit the vector, and subtracting it weakens refusal.

## Pipeline

| Stage | Script | Where it ran |
|---|---|---|
| Seed prompts | `src/build_full_60_scenarios.py` | local |
| Smoke test (hook removal restores baseline exactly) | `src/s00_smoke.py` | Kaggle T4 |
| Curation and scenario-level splits | `src/s01_curate.py` | local |
| Direction extraction and linear probes | `src/s02_extract_probe.py` | Kaggle T4 |
| Dev sweep (layers 14–30 × strengths 0.25/0.5/1.0) | `src/s03_sweep.py` | Kaggle T4 |
| Final generation (A0/A1/A2), capability and latency | `src/s04_final_gen.py` | Kaggle T4 |
| LLM judge | `src/s05_judge_stats.py` | local |
| Toolkit smoke test (CuratorKIT, AuditKIT) | `src/t0_toolkit_smoke.py` | local |
| Toolkit chain: CuratorKIT dataset audit → AuditKIT evaluation | `src/v2_chain.py` | local |
| Scenario-level statistics | `src/v2_stats.py` | local |
| Fail-closed release build with leak gate | `src/build_release.py` | local |
| Hugging Face upload (private first, then public) | `src/publish_hf.py` | local |

The steering hook is in `src/hooks_fallback.py`. Settings are in `configs/experiment.yaml`.

## Reproducing the analysis

The analysis steps run on CPU from frozen artifacts. Raw model completions are private, so `artifacts/raw/` is not in this repository.

```bash
python src/hash_artifacts.py --check   # verify artifacts against artifacts/HASHES.txt
python src/v2_chain.py                  # CuratorKIT + AuditKIT outputs -> artifacts/v2/
python src/v2_stats.py                  # statistics -> artifacts/v2/stats/
python -m pytest tests/                 # statistics and release-gate tests
```

## Release safety

`src/prepush_check.py` runs as a git pre-push hook. It blocks raw completions, the audit sheet, secrets and local paths in every pushed commit. `src/build_release.py` applies the same leak gate to every file in `release/` and fails when its inputs are missing.

## License

- Benchmark prompts and verdicts: CC-BY-4.0, gated on the Hub.
- Steering vectors: CC-BY-NC-4.0, for non-commercial safety research, subject to Cohere Labs' Acceptable Use Policy.

No affiliation with Safaricom, M-Pesa, Airtel Money, Tigo Pesa, KCB, Hustler Fund or any other named service is implied.
