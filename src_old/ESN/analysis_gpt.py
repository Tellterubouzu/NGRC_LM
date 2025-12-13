#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
reports/*.txt に入っている JSON を読み取り、
seed / 最終 PPL / 対応する checkpoint フォルダまでを
run_table.csv にまとめるスクリプト
"""
from pathlib import Path
import re, json, pandas as pd

REPORT_DIR   = Path("./reports_gpt")
#CKPT_DIRROOT = Path("./checkpoint_65536")      # ここは必要に応じて変更
OUT_CSV      = "run_table_gpt.csv"

# ─────────────────────────────────────────────────────────
def extract_ts(name: str) -> str:
    """名前に含まれる 20250719-210847 形式のタイムスタンプを抜き出す"""
    m = re.search(r"(\d{8}-\d{6})", name)
    if m:
        return m.group(1)
    raise ValueError(f"timestamp not found in: {name}")

# def make_ckpt_path(hp: dict, param_millions: float, ts: str) -> str:
#     """命名規則どおりに checkpoint フォルダ名を組み立てる"""
#     sparsity = 1 - (hp["d"] / hp["reservoir_size"])
#     ckpt_name = (
#         f"ESN_ml ({param_millions:.2f}M "
#         f"N{hp['reservoir_size']}_batch_size{hp['local_batch_size']}"
#         f"_seq_len{hp['seq_len']}_sigma_in{hp['sigma_in']}"
#         f"_spectral_radius{hp['spectral_radius']}_sparsity{sparsity}"
#         f"_dropout{hp['dropout']}_r_out{hp['r_out']}_{{'{ts}'}}"
#     )
#     return str(CKPT_DIRROOT / ckpt_name/"checkpoint_step6000_tokens98304000.pt")

# ─────────────────────────────────────────────────────────
records = []
print(f"📁 処理対象ディレクトリ: {REPORT_DIR}")
#print(f"📁 チェックポイントディレクトリ: {CKPT_DIRROOT}")

for rep_path in REPORT_DIR.glob("*.txt"):
    print(f"📄 処理中: {rep_path}")
    with open(rep_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    try:
        hp      = data["hyperparameters"]
        run_id  = data["run_name"]                 # CSV の id にそのまま入れる
        tr_ppl  = data["final_train_perplexity"]
        val_ppl = data["final_val_perplexity"]
        ts      = extract_ts(run_id)
        param_m = data["parameter_count"] / 1e6    # 49.97 など
        msf_ppl = data["test_mean_so_far_ppl_curve"]["2048"]
        
        records.append(
            dict(
                id                     = ts,
                final_train_perplexity = tr_ppl,
                final_val_perplexity   = val_ppl,
                msf_ppl                = msf_ppl,
            )
        )
        
    except Exception as e:
        print(f"❌ エラー発生: {rep_path} - {e}")
        continue

print(f"📊 処理完了: {len(records)} 件のレコードを収集")

if len(records) == 0:
    print("❌ レコードが0件です。DataFrameを作成できません。")
    exit(1)

df = pd.DataFrame(records).sort_values("final_val_perplexity").reset_index(drop=True)
df.to_csv(OUT_CSV, index=False)
print(f"✅ {len(df)} 行を書き出しました → {OUT_CSV}")