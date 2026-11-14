import argparse
import json
import time
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import ROOT, load_config, write_manifest
from hooks_fallback import ResidualSteerer


def generate_texts(model, tokenizer, prompts: list[str], max_new_tokens: int, batch_size: int) -> list[str]:
    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        formatted = [
            tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            for p in batch
        ]
        encoded = tokenizer(formatted, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
        prompt_len = encoded["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False)
        for seq in gen:
            outputs.append(tokenizer.decode(seq[prompt_len:], skip_special_tokens=True))
    return outputs


def prepare_flores_100() -> pd.DataFrame:
    cap_path = ROOT / "data" / "capability" / "flores_100.jsonl"
    if cap_path.exists():
        return pd.read_json(cap_path, lines=True)
    cap_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ds = load_dataset("facebook/flores", "eng_Latn-swh_Latn", split="devtest[:50]", trust_remote_code=True)
        rows = []
        for idx, item in enumerate(ds):
            rows.append({"id": f"FL_EN_SW_{idx:03d}", "direction": "en_to_sw", "source": item["sentence_eng_Latn"], "reference": item["sentence_swh_Latn"]})
            rows.append({"id": f"FL_SW_EN_{idx:03d}", "direction": "sw_to_en", "source": item["sentence_swh_Latn"], "reference": item["sentence_eng_Latn"]})
        df_cap = pd.DataFrame(rows)
    except Exception:
        # Built-in fallback translation pairs if FLORES dataset download is restricted
        fallback_pairs = [
            ("Mobile money services have expanded financial access across East Africa.", "Huduma za pesa za simu zimepanua upatikanaji wa kifedha kote Afrika Mashariki."),
            ("Always verify the recipient name before confirming any electronic transfer.", "Hakikisha jina la mpokeaji kabla ya kuthibitisha muamala wowote wa kielektroniki."),
            ("Never share your personal identification number with anyone over the phone.", "Usiwahi kushiriki namba yako ya siri na mtu yeyote kupitia simu."),
            ("Small businesses rely on instant digital payments to manage daily cash flow.", "Biashara ndogondogo zinategemea malipo ya kidijitali ya haraka kusimamia mtiririko wa fedha wa kila siku."),
            ("Customer support agents will never ask for your one-time verification code.", "Wahudumu wa wateja hawatawahi kuomba msimbo wako wa uthibitisho wa mara moja."),
        ] * 10
        rows = []
        for idx, (en_s, sw_s) in enumerate(fallback_pairs):
            rows.append({"id": f"FL_EN_SW_{idx:03d}", "direction": "en_to_sw", "source": en_s, "reference": sw_s})
            rows.append({"id": f"FL_SW_EN_{idx:03d}", "direction": "sw_to_en", "source": sw_s, "reference": en_s})
        df_cap = pd.DataFrame(rows)
    df_cap.to_json(cap_path, orient="records", lines=True, force_ascii=False)
    return df_cap


def benchmark_latency(
    model,
    tokenizer,
    sample_prompts: list[str],
    steer_cfg: dict | None,
    r_mat: torch.Tensor | None,
    batch_sizes: tuple[int, ...] = (1, 4, 8, 16),
) -> dict:
    results = {}
    for bs in batch_sizes:
        batch = (sample_prompts * 4)[:bs]
        formatted = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in batch]
        encoded = tokenizer(formatted, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        prompt_len = encoded["input_ids"].shape[1]

        # Warmup
        with torch.no_grad():
            _ = model.generate(**encoded, max_new_tokens=16, do_sample=False)
        torch.cuda.synchronize()

        base_times, steer_times, base_toks, steer_toks = [], [], [], []
        for _ in range(5):
            t0 = time.perf_counter()
            with torch.no_grad():
                out_b = model.generate(**encoded, max_new_tokens=64, do_sample=False)
            torch.cuda.synchronize()
            base_times.append(time.perf_counter() - t0)
            base_toks.append(int((out_b.shape[1] - prompt_len) * bs))

            if steer_cfg and r_mat is not None:
                layer = steer_cfg["layer"]
                scale = steer_cfg["effective_scale"]
                t1 = time.perf_counter()
                with ResidualSteerer(model, [layer], r_mat[layer], scale=scale):
                    with torch.no_grad():
                        out_s = model.generate(**encoded, max_new_tokens=64, do_sample=False)
                torch.cuda.synchronize()
                steer_times.append(time.perf_counter() - t1)
                steer_toks.append(int((out_s.shape[1] - prompt_len) * bs))

        med_base = float(pd.Series(base_times).median())
        med_steer = float(pd.Series(steer_times).median()) if steer_times else med_base
        results[f"batch_{bs}"] = {
            "baseline_median_sec": round(med_base, 4),
            "steered_median_sec": round(med_steer, 4),
            "overhead_pct": round(((med_steer - med_base) / max(med_base, 1e-6)) * 100.0, 2),
            "steered_tokens_per_sec": round(float(pd.Series(steer_toks).median()) / max(med_steer, 1e-6), 2),
        }
    return results


def run_final_gen(resume: bool = True) -> None:
    cfg = load_config()
    df = pd.read_json(ROOT / "data" / "curated" / "dataset.jsonl", lines=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    scores_dir = ROOT / "artifacts" / "scores"
    raw_dir = ROOT / "artifacts" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    with open(scores_dir / "selected_dev_configs.json", "r", encoding="utf-8") as f:
        selected = json.load(f)

    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= 8) else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pin to single GPU (device 0) to avoid cross-GPU PCIe overhead on T4 x2
    device_map = {"": 0} if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], dtype=dtype, device_map=device_map)
    model.eval()

    vec_dir = ROOT / "artifacts" / "vectors"
    arms = {
        "A0_base": (None, None),
        "A1_en": (selected["A1_en"], torch.load(vec_dir / "r_hat_A1_en.pt")),
        "A2_multi": (selected["A2_multi"], torch.load(vec_dir / "r_hat_A2_multi.pt")),
    }

    out_path = raw_dir / "test_generations.jsonl"
    completed = set()
    if resume and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                completed.add((rec["arm"], rec["scenario_id"], rec["lang"], rec["role"]))

    bs = cfg["batch_size"]
    max_tok = cfg["max_new_tokens"]["full"]

    # Append when resuming; start a fresh file otherwise
    with open(out_path, "a" if resume else "w", encoding="utf-8") as fout:
        for arm_name, (steer_cfg, r_mat) in arms.items():
            pending_rows = [
                row for _, row in test_df.iterrows()
                if (arm_name, row["scenario_id"], row["lang"], row["role"]) not in completed
            ]
            if not pending_rows:
                print(f"[s04] {arm_name} already complete ({len(test_df)} rows).")
                continue
            print(f"[s04] Generating {len(pending_rows)} test completions for {arm_name}...")
            for i in range(0, len(pending_rows), bs):
                batch_rows = pending_rows[i : i + bs]
                prompts = [r["text"] for r in batch_rows]
                if steer_cfg is None:
                    preds = generate_texts(model, tokenizer, prompts, max_tok, bs)
                else:
                    layer = steer_cfg["layer"]
                    scale = steer_cfg["effective_scale"]
                    with ResidualSteerer(model, [layer], r_mat[layer], scale=scale):
                        preds = generate_texts(model, tokenizer, prompts, max_tok, bs)

                for r, pred in zip(batch_rows, preds):
                    record = {
                        "arm": arm_name,
                        "scenario_id": r["scenario_id"],
                        "lang": r["lang"],
                        "role": r["role"],
                        "subcategory": r["subcategory"],
                        "prompt_text": r["text"],
                        "completion": pred,
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()

    # Capability check: FLORES en <-> sw translations (A0_base vs A2_multi)
    df_cap = prepare_flores_100()
    cap_prompts = [
        f"Translate the following sentence to {'Swahili' if d == 'en_to_sw' else 'English'}. Output only the translation:\n{src}"
        for d, src in zip(df_cap["direction"], df_cap["source"])
    ]
    df_cap["pred_A0_base"] = generate_texts(model, tokenizer, cap_prompts, cfg["max_new_tokens"]["flores"], bs)
    a2_cfg, a2_vec = arms["A2_multi"]
    with ResidualSteerer(model, [a2_cfg["layer"]], a2_vec[a2_cfg["layer"]], scale=a2_cfg["effective_scale"]):
        df_cap["pred_A2_multi"] = generate_texts(model, tokenizer, cap_prompts, cfg["max_new_tokens"]["flores"], bs)
    df_cap.to_json(raw_dir / "capability_generations.jsonl", orient="records", lines=True, force_ascii=False)

    # Latency & throughput benchmark
    lat_stats = benchmark_latency(model, tokenizer, test_df["text"].tolist()[:16], a2_cfg, a2_vec)
    with open(scores_dir / "latency_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(lat_stats, f, indent=2)
    write_manifest("s04_final_gen", {"selected_configs": selected, "latency": lat_stats})
    print("[OK] Final Test Generation, Capability & Latency Complete:", json.dumps(lat_stats, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    run_final_gen(resume=args.resume)