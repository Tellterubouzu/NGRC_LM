#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
reports/*.txt に入っている JSON を読み取り、
seed / 最終 PPL / 対応する checkpoint フォルダまでを
run_table.csv にまとめるスクリプト
"""
from pathlib import Path
import re, json, pandas as pd

REPORT_DIR   = Path("./reports_4096")
CKPT_DIRROOT = Path("./checkpoint_4096")      # ここは必要に応じて変更
OUT_CSV      = "run_table_4096.csv"

# ─────────────────────────────────────────────────────────
def extract_ts(name: str) -> str:
    """名前に含まれる 20250719-210847 形式のタイムスタンプを抜き出す"""
    m = re.search(r"(\d{8}-\d{6})", name)
    if m:
        return m.group(1)
    raise ValueError(f"timestamp not found in: {name}")

def make_ckpt_path(hp: dict, param_millions: float, ts: str) -> str:
    """命名規則どおりに checkpoint フォルダ名を組み立てる"""
    sparsity = 1 - (hp["d"] / hp["reservoir_size"])
    ckpt_name = (
        f"ESN_ml ({param_millions:.2f}M "
        f"N{hp['reservoir_size']}_batch_size{hp['local_batch_size']}"
        f"_seq_len{hp['seq_len']}_sigma_in{hp['sigma_in']}"
        f"_spectral_radius{hp['spectral_radius']}_sparsity{sparsity}"
        f"_dropout{hp['dropout']}_r_out{hp['r_out']}_{{'{ts}'}}"
    )
    return str(CKPT_DIRROOT / ckpt_name/"checkpoint_step6000_tokens98304000.pt")

# ─────────────────────────────────────────────────────────
records = []
for rep_path in REPORT_DIR.glob("*.txt"):
    with open(rep_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    try:
        seed = data["seed"]
    except:
        seed = "unknown"
        continue

    hp      = data["hyperparameters"]
    run_id  = data["run_name"]                 # CSV の id にそのまま入れる
    seed    = data["seed"]
    tr_ppl  = data["final_train_perplexity"]
    val_ppl = data["final_val_perplexity"]
    ts      = extract_ts(run_id)
    param_m = data["parameter_count"] / 1e6    # 49.97 など
    ckpt    = make_ckpt_path(hp, param_m, ts)
    msf_ppl = data["test_mean_so_far_ppl_curve"]["2048"]
    records.append(
        dict(
            id                     = ts,
            seed                   = seed,
            final_train_perplexity = tr_ppl,
            final_val_perplexity   = val_ppl,
            model_path             = ckpt,
            msf_ppl                = msf_ppl,
        )
    )

df = pd.DataFrame(records).sort_values("seed").reset_index(drop=True)
df.to_csv(OUT_CSV, index=False)
print(f"✅ {len(df)} 行を書き出しました → {OUT_CSV}")