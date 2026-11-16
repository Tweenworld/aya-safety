"""Minimal example: apply a released refusal-direction vector to Tiny Aya Earth at inference time.

The vector is added to the output of ONE decoder block at EVERY position (prompt and generated
tokens). This matches the experiment exactly; changing the layer, scale or injection positions
changes the intervention and the reported numbers no longer apply.
"""
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "CohereLabs/tiny-aya-earth"
ARM = "A2_multi"  # or "A1_en"
VECTORS = {"A1_en": "r_hat_A1_en.pt", "A2_multi": "r_hat_A2_multi.pt"}


def load_steering(arm: str) -> tuple[int, float, torch.Tensor]:
    cfg = json.load(open("selected_dev_configs.json", encoding="utf-8"))[arm]
    # effective_scale = strength_mult * mean last-token residual norm at that layer (calibration split)
    r_hat = torch.load(VECTORS[arm])[cfg["layer"]]
    return cfg["layer"], cfg["effective_scale"], r_hat


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="auto")
    layer, scale, r_hat = load_steering(ARM)

    def add_direction(module, inputs, output):  # noqa: ANN001, ARG001
        h = output[0] if isinstance(output, tuple) else output
        h = h + scale * r_hat.to(h.device, h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    handle = model.model.layers[layer].register_forward_hook(add_direction)
    try:
        msgs = [{"role": "user", "content": "How do I recognise a fake M-Pesa reversal request?"}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
        out = model.generate(ids, max_new_tokens=128, do_sample=False)
        print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
    finally:
        # Hooks persist on the module; forgetting this silently steers every later call.
        handle.remove()


if __name__ == "__main__":
    main()
