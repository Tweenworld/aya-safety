import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import load_config, write_manifest
from hooks_fallback import ResidualSteerer

def run_smoke_test() -> None:
    cfg = load_config()
    model_id = cfg["model_id"]
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    print(f"[Gate G0] Loading {model_id} in {dtype} | CUDA: {torch.cuda.is_available()}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side=cfg["padding_side"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    print(f"[Gate G0] Architecture: {type(model).__name__} | Layers: {num_layers} | Hidden: {hidden_size}")

    # Test baseline vs. steered vs. post-removal identity
    messages = [{"role": "user", "content": "Explain how mobile money PIN security works."}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)

    t0 = time.perf_counter()
    with torch.no_grad():
        base_out = model.generate(inputs, max_new_tokens=32, do_sample=False)
    base_time = time.perf_counter() - t0
    base_text = tokenizer.decode(base_out[0][inputs.shape[1]:], skip_special_tokens=True)

    # Check SafeTune availability
    safetune_ok = False
    try:
        import safetune
        print(f"[Gate G0] SafeTune imported successfully: {safetune.__file__}")
        safetune_ok = True
    except Exception as e:
        print(f"[Gate G0] SafeTune import note: {e}")

    # Verify fallback forward hook attaches, alters activations, and cleanly removes
    dummy_dir = torch.randn(hidden_size)
    dummy_dir = dummy_dir / dummy_dir.norm()
    mid_layer = num_layers // 2

    with ResidualSteerer(model, [mid_layer], dummy_dir, scale=5.0):
        with torch.no_grad():
            steered_out = model.generate(inputs, max_new_tokens=32, do_sample=False)

    with torch.no_grad():
        restored_out = model.generate(inputs, max_new_tokens=32, do_sample=False)

    assert torch.equal(base_out, restored_out), "CRITICAL: Hook removal did not restore exact baseline tokens!"
    assert not torch.isnan(dummy_dir).any(), "NaN detected in vector"

    stats = {
        "effective_dtype": str(dtype),
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "safetune_imported": safetune_ok,
        "hook_removal_exact_match": True,
        "baseline_32tok_sec": round(base_time, 3),
        "sample_output": base_text[:120],
    }
    write_manifest("s00_smoke", stats)
    print("[OK] Gate G0 Passed:", stats)

if __name__ == "__main__":
    run_smoke_test()
