import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import ROOT, load_config, write_manifest

def extract_hidden_states(model, tokenizer, texts: list[str], batch_size: int) -> torch.Tensor:
    """Returns [N, num_layers, hidden_size] tensor at the final prompt token."""
    all_layers = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        formatted = [
            tokenizer.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True)
            for t in batch_texts
        ]
        encoded = tokenizer(formatted, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            # Use base transformer directly to avoid allocating LM-head logits VRAM
            out = model.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
        # out.hidden_states has len num_layers + 1 (index 0 is embeddings); take layers 1..L at final token (-1 under left padding)
        layer_stack = torch.stack([h[:, -1, :].detach().cpu().float() for h in out.hidden_states[1:]], dim=1)
        all_layers.append(layer_stack)
    return torch.cat(all_layers, dim=0)

def run_extract_and_probe() -> None:
    cfg = load_config()
    data_path = ROOT / "data" / "curated" / "dataset.jsonl"
    df = pd.read_json(data_path, lines=True)

    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], dtype=dtype, device_map="auto" if torch.cuda.is_available() else None
    )
    model.eval()

    print(f"[s02] Extracting hidden states for {len(df)} rows...")
    H = extract_hidden_states(model, tokenizer, df["text"].tolist(), cfg["batch_size"])
    num_layers = H.shape[1]

    vec_dir = ROOT / "artifacts" / "vectors"
    probe_dir = ROOT / "artifacts" / "probes"
    vec_dir.mkdir(parents=True, exist_ok=True)
    probe_dir.mkdir(parents=True, exist_ok=True)

    # Calibrate split ONLY for refusal directions and layer mean norms
    cal_mask = (df["split"] == "calibrate").values
    df_cal = df[cal_mask].reset_index(drop=True)
    H_cal = H[cal_mask]

    harm_cal = (df_cal["role"] == "harmful").values
    benign_cal = (df_cal["role"] == "benign").values

    # Layer-wise mean activation norms on calibrate
    mean_norms = H_cal.norm(dim=-1).mean(dim=0)  # [num_layers]
    torch.save(mean_norms, vec_dir / "mean_norms.pt")

    # Arm A1: English-only difference-in-means unit refusal direction
    en_harm = ((df_cal["lang"] == "en") & (df_cal["role"] == "harmful")).values
    en_benign = ((df_cal["lang"] == "en") & (df_cal["role"] == "benign")).values
    r_a1 = H_cal[en_harm].mean(dim=0) - H_cal[en_benign].mean(dim=0)
    r_a1 = r_a1 / (r_a1.norm(dim=-1, keepdim=True) + 1e-8)
    torch.save(r_a1, vec_dir / "r_hat_A1_en.pt")

    # Arm A2: Multilingual pooled (en + sw + mix) difference-in-means unit refusal direction
    r_a2 = H_cal[harm_cal].mean(dim=0) - H_cal[benign_cal].mean(dim=0)
    r_a2 = r_a2 / (r_a2.norm(dim=-1, keepdim=True) + 1e-8)
    torch.save(r_a2, vec_dir / "r_hat_A2_multi.pt")

    # Per-language directions for cross-lingual cosine alignment analysis (H2)
    lang_dirs = {}
    for lang in cfg["languages"]:
        lh = ((df_cal["lang"] == lang) & (df_cal["role"] == "harmful")).values
        lb = ((df_cal["lang"] == lang) & (df_cal["role"] == "benign")).values
        d = H_cal[lh].mean(dim=0) - H_cal[lb].mean(dim=0)
        lang_dirs[lang] = d / (d.norm(dim=-1, keepdim=True) + 1e-8)
    torch.save(lang_dirs, vec_dir / "r_hat_per_lang.pt")

    cos_en_sw = torch.nn.functional.cosine_similarity(lang_dirs["en"], lang_dirs["sw"], dim=-1).tolist()
    cos_en_mix = torch.nn.functional.cosine_similarity(lang_dirs["en"], lang_dirs["mix"], dim=-1).tolist()

    # H2 Linear Probes: fit on calibrate (or calibrate+dev), evaluate ONCE on test
    train_mask = df["split"].isin(["calibrate", "dev"]).values
    test_mask = (df["split"] == "test").values
    y_train = (df.loc[train_mask, "role"] == "harmful").astype(int).values
    y_test = (df.loc[test_mask, "role"] == "harmful").astype(int).values

    probe_results = []
    rng = np.random.default_rng(cfg["seeds"]["split"])
    y_train_shuffled = rng.permutation(y_train)

    for layer in range(num_layers):
        X_tr = H[train_mask, layer, :].numpy()
        X_te = H[test_mask, layer, :].numpy()

        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=cfg["seeds"]["split"])
        clf.fit(X_tr, y_train)
        joblib.dump(clf, probe_dir / f"probe_L{layer:02d}.joblib")

        # Shuffled-label control
        clf_shuf = LogisticRegression(C=1.0, max_iter=1000, random_state=cfg["seeds"]["split"])
        clf_shuf.fit(X_tr, y_train_shuffled)

        row_res = {
            "layer": layer,
            "cos_en_sw": round(cos_en_sw[layer], 4),
            "cos_en_mix": round(cos_en_mix[layer], 4),
            "test_auroc_all": round(float(roc_auc_score(y_test, clf.predict_proba(X_te)[:, 1])), 4),
            "test_auroc_shuffled": round(float(roc_auc_score(y_test, clf_shuf.predict_proba(X_te)[:, 1])), 4),
        }
        for lang in cfg["languages"]:
            l_te = (df.loc[test_mask, "lang"] == lang).values
            if l_te.sum() >= 2 and len(np.unique(y_test[l_te])) == 2:
                row_res[f"test_auroc_{lang}"] = round(
                    float(roc_auc_score(y_test[l_te], clf.predict_proba(X_te[l_te])[:, 1])), 4
                )
        probe_results.append(row_res)

    res_df = pd.DataFrame(probe_results)
    res_df.to_csv(probe_dir / "probe_and_cosine_metrics.csv", index=False)

    summary = {
        "num_layers": num_layers,
        "peak_test_auroc_all": float(res_df["test_auroc_all"].max()),
        "peak_auroc_layer": int(res_df.loc[res_df["test_auroc_all"].idxmax(), "layer"]),
        "max_cos_en_sw": float(res_df["cos_en_sw"].max()),
        "max_cos_en_mix": float(res_df["cos_en_mix"].max()),
    }
    write_manifest("s02_extract_probe", summary)
    print("[OK] Extraction & Probes Complete:", json.dumps(summary, indent=2))

if __name__ == "__main__":
    run_extract_and_probe()
