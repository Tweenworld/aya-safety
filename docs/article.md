---
title: "Steering Tiny Aya away from mobile-money fraud in English, Swahili and code-mixed Swahili–English"
authors:
  - user: LeoCleinte
---

# Steering Tiny Aya away from mobile-money fraud in English, Swahili and code-mixed Swahili–English

**TL;DR:** Adding a single refusal direction to the residual stream of [Tiny Aya Earth](https://huggingface.co/CohereLabs/tiny-aya-earth) at inference time reduced usable help on 66 mobile-money fraud prompts from 53.0% to 33.3% (English direction) and 31.8% (multilingual direction, both p = 0.004). Multilingual calibration performed comparably to English-only calibration (p = 1.0) while increasing refusal phrasing in matched legitimate answers from 1/66 to 11/66. These gains shrink to −13.6 points (p = 0.09) if partial help is excluded, and specificity remains untested without a random-direction control.

Artifacts: [steering vectors & model collection](https://huggingface.co/collections/LeoCleinte/tiny-aya-earth-swahili-and-sheng-safety-steering-6afab528fb5ebfc572c73403) · [benchmark, verdicts & dataset collection](https://huggingface.co/collections/LeoCleinte/east-african-mobile-money-fraud-safety-audit-6afab6956a5b1af4d9c43752) · [code](https://github.com/Tweenworld/aya-safety) (tag `results-v1-frozen`)

## Why I chose this problem

Mobile money is everyday financial infrastructure in Kenya, and SMS-based social engineering—fake M-Pesa confirmation messages, "wrong number" refund pleas, and fake Fuliza or loan-limit alerts—is a constant nuisance that people around me deal with regularly. As open-weights multilingual models like Tiny Aya Earth become small enough to run locally or on-device for regional customer support, I wanted to test whether a lightweight safety intervention could stop them from generating usable scam templates in Swahili and code-mixed Swahili–English without refusing legitimate anti-fraud questions, such as asking how to spot a forged M-Pesa SMS.

## Setup

**Benchmark.** The dataset contains 60 scenarios across three languages (`en`, `sw`, `mix`), each written as a harmful request and a matched benign twin (360 prompts total). For example, a harmful prompt asks for a realistic fake confirmation SMS to walk out with goods, while its benign twin asks for a sample fake SMS to train shopkeepers to spot forgeries. Scenarios are split 25 for calibration, 13 for dev, and 22 for test so no scenario crosses splits. The prompts were written as literals in generator scripts. The Swahili and code-mixed text has **not** been verified by a native speaker, and the exact code-mixed variety has not been reviewed. The `origin: human_seed`, `reviewer` and `confirmed: true` fields in the JSONL file are constants written by those scripts, not a record of human writing or review (see the dataset card).

**Directions and selection.** On the calibration split, I extracted the last prompt token's residual activation at each layer and computed the normalized difference `mean(harmful) − mean(benign)` using English prompts alone (**A1**) or all three languages pooled (**A2**). During generation, `scale × r̂` is added to one decoder block's output across all token positions, where `scale` equals a strength multiplier times the layer's mean residual norm. Sweeping layers 14–30 and strengths {0.25, 0.5, 1.0} on the dev split selected the lowest strength (0.25×) for both arms: A1 at layer 26 (scale 18.3) and A2 at layer 22 (scale 14.1).

**Evaluation.** I generated greedy 256-token completions on the 132 test prompts per arm. An LLM judge scored harmful outputs for usable fraud help and benign outputs as helpful, partly helpful, or refused; 372 of 396 verdicts came from `nemotron-3.5-super-vl:free` and 24 (baseline only) from `gemini-3.8-flash`.

## Metrics

I tracked three outcomes: **ASR** (whether a harmful reply provides usable fraud help, catching cases where a model adds a disclaimer before writing the scam), **judged over-refusal** on benign twins, and **lexical refusal phrases** in benign answers ("I can't help", "siwezi", "samahani", excluding topic words like "fraud" and "scam").

Because a scenario's three translations share underlying structure, every test treats the 22 test scenarios as the unit of analysis using paired scenario-level differences, scenario-bootstrap 95% confidence intervals, Clopper–Pearson proportion intervals, and multiple-testing p-value adjustments across the main comparisons.[^stats]

[^stats]: Exact paired sign-flip tests with Holm correction applied separately across the two main ASR comparisons (A1 vs base, A2 vs base) and the two benign over-refusal checks (A1 vs base, A2 vs base).

## Results

| Metric (of 66) | Base | + A1 English | + A2 multilingual |
|---|---|---|---|
| **Usable help on harmful prompts** | 35 (53.0%) [40.3, 65.4] | 22 (33.3%) [22.2, 46.0] | 21 (31.8%) [20.9, 44.4] |
| Judged refusal of benign prompts | 0 (0.0%) [0.0, 5.4] | 2 (3.0%) [0.4, 10.5] | 5 (7.6%) [2.5, 16.8] |
| Refusal phrases in benign answers | 1 (1.5%) [0.0, 8.2] | 10 (15.2%) [7.5, 26.1] | 11 (16.7%) [8.6, 27.9] |

*95% Clopper–Pearson intervals, in percent.*

| Comparison | Difference | 95% CI | Scenarios changed | p (Holm) |
|---|---|---|---|---|
| ASR: A1 vs base | −19.7 pts | [−30.3, −10.6] | 10/22 | 0.004 |
| ASR: A2 vs base | −21.2 pts | [−31.8, −10.6] | 13/22 | 0.004 |
| Over-refusal: A1 vs base | +3.0 pts | [0.0, +7.6] | 2/22 | 0.50 |
| Over-refusal: A2 vs base | +7.6 pts | [+1.5, +15.2] | 4/22 | 0.25 |
| ASR: A2 vs A1 *(secondary)* | −1.5 pts | [−9.1, +4.5] | 5/22 | 1.00 (unadjusted) |
| Refusal phrases: A2 vs base *(secondary)* | +15.2 pts | [+4.5, +28.8] | 8/22 | 0.047 (unadjusted) |

**Key takeaways**

1. **Both steering directions reduce judged usable help.** Removing the 12 harmful and 12 benign prompts whose baseline answers Gemini scored (from all three arms), or the two out-of-vocabulary judge labels, yields Holm p = 0.008 for both arms.
2. **Multilingual calibration performed similarly to English-only calibration.** Per language, usable help changed from 10 to 6 (`en`), 12 to 6 (`sw`), and 13 to 10 (A1) or 9 (A2) (`mix`) out of 22 scenarios each, with code-mixed prompts showing the smallest drop.
3. **Benign refusal phrasing rose even when the judge scored answers as helpful.** With 22 scenarios, the over-refusal test only reliably detects shifts of 6 to 10 points, while lexical refusal phrases in benign replies rose roughly tenfold. One untested hypothesis is that my judge's instruction to score protective content as helpful absorbed hedged refusals.
4. **Effect size depends on how partial help is counted.** Counting only `full_help` as usable reduces both drops to 13.6 points (Holm p = 0.09).

## Limitations

- **No random-direction control.** Without testing random vectors of matching norm at layers 22 and 26, I cannot separate direction-specific steering from a generic activation perturbation.
- **Judge and selection constraints.** One free-tier LLM judge scored completions while seeing each prompt's role, without human validation; my primary Gemini endpoint failed after 3 batches after timeouts, service unavailability, and rate-limit errors printed to the console (not saved to an artifact log), failing over to `nemotron-3.5-super-vl:free`. Dev selection also relied on a 32-token lexical regex that matched topic words like "fraud" and "scam", pushing all 6 passing configurations to the lowest tested strength (0.25× norm).
- **No capability claim.** The `facebook/flores` load failed (the error was caught and not logged), so the translation check fell back to 5 repeated in-domain sentence pairs.
- **Truncation and lexical probes.** Completions stopped at 256 tokens, leaving any post-cap compliance unmeasured. Linear probes separated harmful from benign prompts at AUROC ≥ 0.99 from layer 4 onward (0.91–0.98 at layers 0–3), which reflects wording differences rather than a verified cross-lingual safety circuit.
- **Scope and security boundary.** Evaluation covers single-turn, non-adversarial prompts only. Activation steering is an inference-time behavioral nudge rather than a security boundary: anyone with model weights can omit the vector, and subtracting it weakens refusal.
- **Latency.** T4 overhead measured 1.0% to 3.1% across batch sizes 1 to 16 (64 new tokens, 5 repeats) without fixed output lengths, so treat it as indicative.

## Pipeline, toolkits, and disclosure

**Systems and evaluation engineering:** To avoid cross-GPU PCIe transfers from splitting the `float16` model across two T4 GPUs, I pinned final test generation to a single T4 (`device_map={"": 0}`) and ran inference at `batch_size=16`, with 1.0%–3.1% steering latency overhead. For LLM judging, I packed up to 8 prompt–response pairs per request (50 batches for 396 completions) to stay inside free-tier request quotas.

All results come from frozen, SHA-256-hashed artifacts (`tag: results-v1-frozen`). **CuratorKIT 0.1.27** audited the 360 prompts (schema gate and MinHash near-duplicate check at 0.85, finding 0 duplicates), while **AuditKIT 1.1.11** reproduced every headline count from the frozen judge verdicts and produced its own metric tables and paired comparisons; the intervals and p-values above come from a separate scenario-level statistics script with 21 unit tests. Steering and judging ran via custom PyTorch forward hooks and LiteLLM (`SafeTune`'s `RefusalDirectionModel(mode="steer")` implements the equivalent hook operation but was not executed). A shared leak gate scans every release file and commit so raw completions, keys, and local paths remain private.

**Responsible release:** The benchmark is gated (`CC-BY-4.0`) and raw model completions are withheld because some baseline answers contain usable scam content. The steering vectors are released under `CC-BY-NC-4.0` for non-commercial safety research subject to Cohere Labs' Acceptable Use Policy. No affiliation with Safaricom, M-Pesa, Airtel Money, Tigo Pesa, KCB, Hustler Fund, or any named service is implied.

**AI tool disclosure:** I used an AI coding and research assistant to help scaffold pipeline scripts, debug runtime and API failover issues, structure statistical checks, and edit documentation; all experimental runs, verification checks, and repository releases were executed and reviewed by me.

## Next steps

1. Run a multi-seed random-direction control at layers 22 and 26.
2. Replace regex-based dev selection with judge-based selection over a finer strength grid.
3. Conduct an independent native-speaker audit of the Swahili and code-mixed prompts and a sample of judge verdicts.
4. Evaluate general capability retention on Parquet-hosted FLORES-200 slices.
5. Cross-check the forward hook directly against SafeTune's RefusalDirectionModel.
6. Expand the v2 benchmark to cover additional regional advance-fee patterns, such as port and customs parcel clearance scams (impersonating customs or courier agents requesting mobile-money fast-track fees).