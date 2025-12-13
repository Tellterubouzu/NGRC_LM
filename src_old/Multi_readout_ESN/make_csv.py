#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
reports/*.txt の JSON から
seed / PPL / checkpoint の実在パス を run_table.csv にまとめる
（mlrmro_esn.py の命名に追従し、ディレクトリ/ファイルは実際に探索）
"""
from pathlib import Path
import re, json, math
import pandas as pd

REPORT_DIR   = Path("./reports")
CKPT_DIRROOT = Path("./checkpoint")
OUT_CSV      = "run_table.csv"   # metrics.py のデフォルトに合わせる

# ─────────────────────────────────────────────────────────
def extract_ts(s: str) -> str:
    """ESN_mlo (...)_{'20250719-210847'} のような名前から 20250719-210847 を抽出"""
    m = re.search(r"(\d{8}-\d{6})", s)
    if not m:
        raise ValueError(f"timestamp not found in: {s}")
    return m.group(1)

def list_pt_files(d: Path) -> list[Path]:
    return sorted(d.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)

def score_dir(name: str, hp: dict, param_millions: float, ts: str) -> int:
    """候補ディレクトリの名前一致度にスコアを付ける（高いほど良い）"""
    sc = 0
    if ts in name: sc += 5                        # タイムスタンプ一致を最重視
    # なるべく厳密に寄せる
    if f"N{hp.get('reservoir_size')}" in name: sc += 3
    if f"_nro{hp.get('num_readouts')}" in name: sc += 2
    if f"_mro{hp.get('num_readouts')}" in name: sc += 2  # 念のため
    if f"_batch_size{hp.get('local_batch_size')}" in name: sc += 1
    if f"_seq_len{hp.get('seq_len')}" in name: sc += 1
    if f"_sigma_in{hp.get('sigma_in')}" in name: sc += 1
    if f"_spectral_radius{hp.get('spectral_radius')}" in name: sc += 1
    if f"_dropout{hp.get('dropout')}" in name: sc += 1
    if f"_r_out{hp.get('r_out')}" in name: sc += 1
    # パラメータ数は丸め誤差を考慮して文字列一致を弱めに
    if f"{param_millions:.2f}M" in name: sc += 2
    return sc

def find_ckpt_dir_and_file(ts: str, hp: dict, param_millions: float) -> tuple[str, str]:
    """
    ./checkpoint 配下から、タイムスタンプ ts を含む最も尤もらしいディレクトリを探し、
    その中の最新 .pt を返す。見つからなければ ("","")。
    """
    if not CKPT_DIRROOT.exists():
        return "", ""

    # まず直下のディレクトリを候補化
    dirs = [d for d in CKPT_DIRROOT.iterdir() if d.is_dir()]
    if not dirs:
        return "", ""

    # ts を含むものを優先、無ければ全体からスコア上位を採用
    scored = []
    for d in dirs:
        sc = score_dir(d.name, hp, param_millions, ts)
        if sc > 0:
            scored.append((sc, d))
    if not scored:
        # それでも無い場合は全ディレクトリをスコアリング
        scored = [(score_dir(d.name, hp, param_millions, ts), d) for d in dirs]

    # スコア降順で探索し、最初に .pt が見つかった場所を使用
    for _, d in sorted(scored, key=lambda x: x[0], reverse=True):
        pts = list_pt_files(d)
        if pts:
            return str(d), str(pts[0])

    # 最後の手段：全体を再帰検索（重いので最後に）
    pts_all = sorted(CKPT_DIRROOT.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in pts_all:
        d = p.parent
        if score_dir(d.name, hp, param_millions, ts) > 0:
            return str(d), str(p)

    return "", ""

def pick_msf_ppl(curve: dict, k: int = 2048) -> float | None:
    if not isinstance(curve, dict):
        return None
    # キーは文字列想定
    key = str(k)
    v = curve.get(key)
    try:
        return float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else None
    except Exception:
        return None

# ─────────────────────────────────────────────────────────
records = []
for rep_path in sorted(REPORT_DIR.glob("*.txt")):
    with open(rep_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 必須フィールド
    if "seed" not in data or "hyperparameters" not in data or "parameter_count" not in data:
        print(f"[skip] malformed report: {rep_path.name}")
        continue

    seed    = data["seed"]
    run_id  = data.get("run_name", rep_path.stem)      # wandb の run 名
    hp      = data["hyperparameters"]
    tr_ppl  = data.get("final_train_perplexity")
    val_ppl = data.get("final_val_perplexity")
    param_m = float(data["parameter_count"]) / 1e6

    # タイムスタンプ抽出（run_name 優先、ダメならレポートファイル名）
    try:
        ts = extract_ts(run_id)
    except Exception:
        ts = extract_ts(rep_path.stem)

    # 実在する checkpoint を探索
    model_dir, model_path = find_ckpt_dir_and_file(ts, hp, param_m)

    # mean-so-far PPL（@2048 を優先、無ければ None）
    msf_curve = data.get("test_mean_so_far_ppl_curve", {})
    msf_ppl   = pick_msf_ppl(msf_curve, k=2048)

    records.append(
        dict(
            id                     = ts,
            seed                   = seed,
            final_train_perplexity = tr_ppl,
            final_val_perplexity   = val_ppl,
            model_dir              = model_dir,
            model_path             = model_path,
            msf_ppl                = msf_ppl,
        )
    )

df = pd.DataFrame(records).sort_values(["seed","id"]).reset_index(drop=True)
df.to_csv(OUT_CSV, index=False)
print(f"✅ {len(df)} 行を書き出しました → {OUT_CSV}")
