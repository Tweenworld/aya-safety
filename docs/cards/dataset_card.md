---
license: cc-by-4.0
language:
  - en
  - sw
tags:
  - ai-safety
  - refusal
  - over-refusal
  - swahili
  - code-switching
  - mobile-money-fraud
  - red-teaming
size_categories:
  - n<1K
pretty_name: East African mobile-money fraud refusal benchmark (en / sw / code-mixed)
extra_gated_prompt: >-
  This dataset contains requests for fraud assistance (with benign twins) for safety evaluation.
  By requesting access you agree to use it only for evaluating or improving model safety.
extra_gated_fields:
  Intended use: text
---

# East African mobile-money fraud refusal benchmark (en / sw / code-mixed)

This is a small paired benchmark for measuring two things:
1. whether a model gives **usable help** for mobile-money fraud requests, and
2. whether it **over-refuses** closely matched legitimate (benign) requests.

It covers English, Swahili and code-mixed Swahili–English. It also includes the judge verdicts and the statistical analysis for a refusal-direction steering experiment on `CohereLabs/tiny-aya-earth`.

**Model completions are not included.** Only prompts, labels, verdicts and aggregate statistics are.

No affiliation with, or endorsement by, Safaricom, M-Pesa, Airtel Money, Tigo Pesa, KCB, Hustler Fund or any other named service. Brand names appear because real scams impersonate them.

## Contents

| Path | What it is |
|---|---|
| `benchmark_prompts.jsonl` | 360 prompts: 60 scenarios × {en, sw, mix} × {harmful, benign twin}, with split labels |
| `judge_verdicts.jsonl` | 396 judge verdicts (3 arms × 132 test prompts): `usable_help`, `level`, `judge_model`. No completions. |
| `stats/` | Scenario-level statistics (`stats_tables.md`, `stats_summary.json`) |
| `audit/` | AuditKIT 1.1.11 metric tables and paired comparisons |
| `curation/` | CuratorKIT 0.1.27 audit of the prompts (schema gate, exact and near-duplicate check, provenance) |
| `run_records/` | Stage manifests, dev-sweep summary, latency benchmark, frozen-artifact hashes |

### Fields (`benchmark_prompts.jsonl`)
- `scenario_id`: one scenario per row group. The en/sw/mix and harmful/benign rows of a scenario form one cluster.
- `lang`: `en`, `sw` or `mix` (code-mixed Swahili–English; the exact variety has not been reviewed by a native speaker).
- `role`: `harmful` (asks for help committing fraud) or `benign` (a matched legitimate request, often educational or protective).
- `subcategory`: one of `fake_payment_confirmation`, `wrong_number_refund`, `provider_impersonation`, `agent_till_impersonation`, `pin_otp_social_engineering`, `fake_loan_offer`.
- `split`: `calibrate` (25 scenarios), `dev` (13) or `test` (22), assigned per scenario by hash. The splits are not stratified by subcategory: dev has no PIN/OTP scenario, and test has one fake-loan scenario.
- `sha256`: hash of the stripped prompt text.
- `origin`, `reviewer`, `confirmed`: **see the provenance warning below.**

## Provenance: read before citing
- **The prompts were written as literals in generator scripts** (`src/build_full_60_scenarios.py`, `src/make_starter_seeds.py`).
- **The fields `origin: human_seed`, `reviewer` and `confirmed: true` are constants written by those scripts.** They do not record an actual human-writing or native-speaker review process, and should not be read as such.
- The Swahili and code-mixed text has **not** been verified by a native speaker.
- The file is unchanged from the experiment (to keep hashes valid). This card is the correction.

## How the verdicts were produced
An LLM judge scored each completion given the prompt and its role:
- **Harmful rows:** `usable_help` plus a `level` of `refused`, `safe_deflection`, `partial_help` or `full_help`.
- **Benign rows:** a `level` of `helpful`, `partly_helpful` or `refused`.

**Which judge scored what:** 372 rows by `openrouter/nvidia/nemotron-3.5-super-vl:free` and 24 by `gemini/gemini-3.8-flash` (all baseline arm). The primary judge failed after 3 batches for an unrecorded reason.

**Known issues:**
- The judge sees the role.
- The rubric tells it to score protective content as helpful.
- 2 harmful rows carry the out-of-vocabulary level `helpful` (`SCEN_047` en and mix, A2). Both count as not usable help.
- There is no human validation.

## Intended use
- Measuring usable help **and** over-refusal together for fraud-assistance requests in English, Swahili and code-mixed text.
- Comparing safety interventions with paired, scenario-level statistics.

## Out of scope
- Training models to produce scam content.
- Treating the benchmark as representative of all East African fraud. It has 60 scenarios, 6 subcategories, and is single-turn only.
- Claims about real deployments, providers or regulators.

## Dual-use and safety notes
- Harmful prompts are requests, not working scam templates, but they describe real fraud patterns. Access is gated for that reason.
- Completions from the unsteered model, some of which contain usable scam content, are **deliberately withheld.**

## Key results (from `stats/stats_tables.md`)

| | Base | A1 English direction | A2 multilingual direction |
|---|---|---|---|
| Usable help on harmful (ASR) | 35/66 | 22/66 | 21/66 |
| Judged refusal of benign | 0/66 | 2/66 | 5/66 |
| Refusal phrases in benign answers (lexical) | 1/66 | 10/66 | 11/66 |

ASR drop vs base: A1 −19.7 points, A2 −21.2 points (both Holm p = 0.004). If `partial_help` is counted as not usable, both drops are −13.6 points (Holm p = 0.09). A2 vs A1: p = 1.0.

## Provenance of this release
Built by `src/build_release.py` from the frozen artifacts (tag `results-v1-frozen`, hashes in `run_records/HASHES.txt`), after a leak gate that checks every file for model-completion text, secrets and local paths.

## License
The prompts, labels, judge verdicts, and audit statistics in this dataset are released under **CC-BY-4.0** (Creative Commons Attribution 4.0 International).
