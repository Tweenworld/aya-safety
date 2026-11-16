---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-earth
tags:
  - activation-steering
  - refusal-direction
  - ai-safety
  - swahili
  - code-switching
  - mobile-money-fraud
language:
  - en
  - sw
---

# Refusal-direction steering vectors for Tiny Aya Earth (East African mobile-money fraud)

These are **inference-time steering vectors, not model weights.** Adding one of them to the residual stream of [`CohereLabs/tiny-aya-earth`](https://huggingface.co/CohereLabs/tiny-aya-earth) during generation reduced how often the model gave **usable help** for mobile-money fraud requests in English, Swahili and code-mixed Swahili–English. The test set has 22 scenarios. The effect is real but modest and fragile, and its specificity to *this* direction is **untested**. Read Limitations before using it.

No affiliation with, or endorsement by, Safaricom, M-Pesa, Airtel Money, Tigo Pesa, KCB, Hustler Fund or any other named service. Brand names appear only because real scams impersonate them.

## What is in this repo

| File | Contents |
|---|---|
| `r_hat_A1_en.pt` | Unit difference-of-means direction for each layer, shape `[36, 2048]`, estimated from **English** calibration prompts only |
| `r_hat_A2_multi.pt` | Same, estimated from **pooled en + sw + code-mixed** calibration prompts |
| `r_hat_per_lang.pt` | Per-language directions (dict `en`/`sw`/`mix`), used only for the cosine analysis |
| `mean_norms.pt` | Mean last-prompt-token residual norm per layer on the calibration split, shape `[36]` |
| `selected_dev_configs.json` | Layer and scale chosen on the dev split (see "How selection was done") |
| `experiment.yaml` | The configuration of the run that produced these files |
| `steer_example.py` | A minimal loader that applies a vector with a forward hook |

**Selected configurations:**
- A1: layer 26, scale 18.29 (= 0.25 × mean norm).
- A2: layer 22, scale 14.14 (= 0.25 × mean norm).

Layer `l` means the output of decoder block `l` (`hidden_states[l + 1]` in Transformers).

## How to use

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "CohereLabs/tiny-aya-earth"
tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map="auto")

layer, scale = 22, 14.1381                     # A2 selection from selected_dev_configs.json
r_hat = torch.load("r_hat_A2_multi.pt")[layer]  # unit vector, shape [2048]

def add_direction(module, inputs, output):
    # Same operation as SafeTune's RefusalDirectionModel(mode="steer") and the project's
    # ResidualSteerer: add scale * r_hat to the block output at every position.
    h = output[0] if isinstance(output, tuple) else output
    h = h + scale * r_hat.to(h.device, h.dtype)
    return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

handle = model.model.layers[layer].register_forward_hook(add_direction)
try:
    msgs = [{"role": "user", "content": "How do I recognise a fake M-Pesa reversal request?"}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    print(tok.decode(model.generate(ids, max_new_tokens=128, do_sample=False)[0][ids.shape[1]:]))
finally:
    handle.remove()  # always detach; a leaked hook silently steers every later call
```

## Results (test split: 22 scenarios × 3 languages = 66 harmful and 66 benign prompts)

Greedy decoding, 256 new tokens. Judged mainly by `nemotron-3.5-super-vl:free` through OpenRouter (372 of 396 rows; the other 24 by `gemini-3.8-flash`, all in the baseline arm). Unit of analysis: the scenario. Source: `artifacts/v2/stats/stats_tables.md` in the GitHub repo.

| Metric | Base | + A1 (English) | + A2 (multilingual) |
|---|---|---|---|
| **Usable help on harmful prompts (ASR)** | 35/66 (53.0%) | 22/66 (33.3%) | 21/66 (31.8%) |
| Judged refusal of benign prompts | 0/66 (0.0%; 95% CI up to 5.4%) | 2/66 (3.0%) | 5/66 (7.6%) |
| Refusal phrases in benign answers (lexical) | 1/66 (1.5%) | 10/66 (15.2%) | 11/66 (16.7%) |

| Comparison | Difference | 95% CI (scenario bootstrap) | p (exact sign-flip, Holm) |
|---|---|---|---|
| ASR, A1 vs base | −19.7 pts | [−30.3, −10.6] | 0.004 |
| ASR, A2 vs base | −21.2 pts | [−31.8, −10.6] | 0.004 |
| Over-refusal, A1 vs base | +3.0 pts | [0.0, +7.6] | 0.50 |
| Over-refusal, A2 vs base | +7.6 pts | [+1.5, +15.2] | 0.25 |
| ASR, A2 vs A1 (secondary) | −1.5 pts | [−9.1, +4.5] | 1.00 (unadjusted) |

**How to read this:**
- Both directions lower judged usable help by about 20 points.
- **Multilingual calibration did not beat English-only** (p = 1.0).
- The judge's over-refusal increase is not significant. But with 22 scenarios the smallest reliably detectable change is about 6 to 10 points, and the lexical refusal-phrase rate in benign answers rose about tenfold (secondary, unadjusted p ≈ 0.05). So expect more hedging and refusal language on legitimate anti-fraud questions.
- **The result is sensitive to how partial help is counted.** If `partial_help` is not counted as usable, both drops shrink to −13.6 points (Holm p = 0.09).

## How selection was done (and why it is weak)
Layer and strength were chosen on 13 dev scenarios using a **32-token lexical refusal regex**, not the judge. The regex also matched topic words such as "fraud" and "scam". Only 6 of 54 configurations passed the over-refusal gate, all at the lowest strength tried (0.25 × norm). The optimum is therefore unknown. Treat the selected scales as a starting point.

## Threat model and dual use
- **In scope:** a non-adversarial user sends direct single-turn requests to a model served *with* the vector applied.
- **Out of scope and untested:** jailbreaks, multi-turn escalation, obfuscation, persona framing.
- **Not a security boundary.** Anyone with the weights can omit the vector. **Subtracting it is a known way to weaken refusal.** That is why these artifacts are documented as a defensive research artifact only.

## Limitations
- **Specificity is untested.** No random-direction control was run, so the effect may be a generic perturbation of this size rather than this direction.
- **Judge:** a single free-tier LLM judge that sees each prompt's harmful/benign role. Its rubric instructs that protective content "MUST" be scored helpful. There is no human validation, and its Swahili and code-mixed accuracy is unknown.
- **Capability retention is not established.** The translation check fell back to 5 repeated in-domain sentences, so no capability claim is made.
- **Truncation:** completions were capped at 256 tokens. Any compliance after the cap is invisible to the judge.
- **Latency overhead** measured on a T4 was 1.0 to 3.1% across batch sizes 1 to 16 (64 tokens, 5 repeats). That benchmark did not fix output length, so treat it as indicative.
- **The probes and cosine analyses** show lexical separability of the prompts. They are not evidence of a cross-lingual "safety representation".
- **Small sample:** 22 test scenarios; per-language cells have 22 rows each and are descriptive only.

## Provenance
- **Code:** <https://github.com/Tweenworld/aya-safety>, tag `results-v1-frozen`. Frozen inputs are hashed in `artifacts/HASHES.txt`.
- **Toolkits that ran on project data:**
  - **CuratorKIT 0.1.27** (dataset audit).
  - **AuditKIT 1.1.11** (evaluation tables and paired comparisons; it aggregates the judge verdicts and is not itself the judge).
  - Steering used the project's own hook. SafeTune's `RefusalDirectionModel(mode="steer")` implements the same operation but was **not** run.

## License
These steering vectors were extracted from [`CohereLabs/tiny-aya-earth`](https://huggingface.co/CohereLabs/tiny-aya-earth) and are distributed under **CC-BY-NC-4.0** (Creative Commons Attribution-NonCommercial 4.0) for non-commercial safety research only. Downstream users must also adhere to Cohere Labs' Acceptable Use Policy.
