#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# JSONっぽいテキストを頑健に読む
# ----------------------------
def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def load_jsonlike_file(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = _strip_code_fences(text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError(f"JSON object not found in: {path}")
    blob = text[start : end + 1]
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        blob2 = re.sub(r",\s*([}\]])", r"\1", blob)  # 末尾カンマ除去
        return json.loads(blob2)


def flatten_run(obj: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    row["file"] = str(file_path)
    row["run_name"] = obj.get("run_name")
    # 代表的な評価値
    for k in [
        "final_train_perplexity",
        "final_val_perplexity",
        "final_train_loss",
        "final_val_loss",
        "parameter_count",
        "training_time_sec",
        "seed",
    ]:
        if k in obj:
            row[k] = obj.get(k)

    hp = obj.get("hyperparameters") or {}
    if isinstance(hp, dict):
        for k, v in hp.items():
            col = f"hp.{k}"
            if isinstance(v, (dict, list)):
                row[col] = json.dumps(v, ensure_ascii=False, sort_keys=True)
            else:
                row[col] = v
    return row


# ----------------------------
# 集計＆可視化
# ----------------------------
def bootstrap_ci_median(x: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    # 小標本でも使える「中央値」のブートストラップCI
    rng = np.random.default_rng(seed)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(x, size=len(x), replace=True)
        boots.append(np.median(samp))
    lo = np.quantile(boots, alpha / 2)
    hi = np.quantile(boots, 1 - alpha / 2)
    return float(lo), float(hi)


def make_heatmap(values_pivot: pd.DataFrame, count_pivot: pd.DataFrame, title: str, out_path: Path) -> None:
    # rows: poly, cols: lag
    vals = values_pivot.to_numpy(dtype=float)
    counts = count_pivot.reindex(index=values_pivot.index, columns=values_pivot.columns).to_numpy(dtype=float)

    masked = np.ma.masked_invalid(vals)

    fig, ax = plt.subplots(figsize=(1.2 + 0.9 * values_pivot.shape[1], 1.2 + 0.6 * values_pivot.shape[0]))
    im = ax.imshow(masked, aspect="auto")  # colormapはデフォルト
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    ax.set_title(title)
    ax.set_xlabel("ngrc_lag")
    ax.set_ylabel("ngrc_poly_degree")

    ax.set_xticks(np.arange(values_pivot.shape[1]))
    ax.set_xticklabels([str(c) for c in values_pivot.columns])
    ax.set_yticks(np.arange(values_pivot.shape[0]))
    ax.set_yticklabels([str(i) for i in values_pivot.index])

    # 各セルに値とnを注記
    for i in range(values_pivot.shape[0]):
        for j in range(values_pivot.shape[1]):
            v = vals[i, j]
            n = counts[i, j]
            if np.isfinite(v) and np.isfinite(n) and n > 0:
                ax.text(j, i, f"{v:.1f}\n(n={int(n)})", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, default=None, help="*_report.txt があるディレクトリ（指定するなら）")
    ap.add_argument("--pattern", type=str, default="*_report.txt", help="input_dir指定時のglob")
    ap.add_argument("--input_csv", type=str, default=None, help="runs_parsed.csv を使う場合はこちら")
    ap.add_argument("--target", type=str, default="final_train_perplexity", help="最適化したい指標（小さいほど良い想定）")

    ap.add_argument("--out_dir", type=str, default="./lag_poly_out")
    ap.add_argument("--where", type=str, default=None, help='pandas query で条件指定（例: "hp.ngrc_d_model == 512"）')
    ap.add_argument("--min_count", type=int, default=2, help="セル採用の最小サンプル数")
    ap.add_argument("--agg", choices=["median", "mean"], default="median", help="セル値に使う集計")
    ap.add_argument("--per_d_model", action="store_true", help="d_modelごとにも同様の表/図を出す")
    ap.add_argument("--bootstrap_ci", action="store_true", help="セル中央値のブートストラップCIを出す（遅い）")
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load ----
    if args.input_csv:
        df = pd.read_csv(args.input_csv)
    else:
        if not args.input_dir:
            raise SystemExit("Either --input_csv or --input_dir must be provided.")
        in_dir = Path(args.input_dir)
        files = sorted(in_dir.glob(args.pattern))
        if not files:
            raise SystemExit(f"No files matched: {in_dir} / {args.pattern}")
        rows = []
        errors = []
        for fp in files:
            try:
                obj = load_jsonlike_file(fp)
                rows.append(flatten_run(obj, fp))
            except Exception as e:
                errors.append((str(fp), str(e)))
        df = pd.DataFrame(rows)
        if errors:
            (out_dir / "parse_errors.txt").write_text("\n".join([f"{f}\t{m}" for f, m in errors]), encoding="utf-8")

    # 必要列
    need = ["hp.ngrc_lag", "hp.ngrc_poly_degree", args.target]
    for c in need:
        if c not in df.columns:
            raise SystemExit(f"Missing column: {c}")

    # 型変換
    df["hp.ngrc_lag"] = pd.to_numeric(df["hp.ngrc_lag"], errors="coerce")
    df["hp.ngrc_poly_degree"] = pd.to_numeric(df["hp.ngrc_poly_degree"], errors="coerce")
    df[args.target] = pd.to_numeric(df[args.target], errors="coerce")

    df = df.dropna(subset=["hp.ngrc_lag", "hp.ngrc_poly_degree", args.target]).copy()

    # 条件フィルタ
    if args.where:
        df = df.query(args.where).copy()

    # 保存（フィルタ後のデータ）
    df.to_csv(out_dir / "filtered_runs.csv", index=False, encoding="utf-8")

    # ---- 2D集計 ----
    g = df.groupby(["hp.ngrc_poly_degree", "hp.ngrc_lag"])[args.target]
    summary = g.agg(
        count="size",
        mean="mean",
        median="median",
        std="std",
    ).reset_index()

    # セルの採用基準
    summary_ok = summary[summary["count"] >= args.min_count].copy()
    summary.to_csv(out_dir / "lag_poly_summary_all.csv", index=False, encoding="utf-8")
    summary_ok.to_csv(out_dir / "lag_poly_summary_min_count.csv", index=False, encoding="utf-8")

    # ---- bestの抽出（観測ベース）----
    sort_key = args.agg
    top = summary_ok.sort_values(sort_key, ascending=True).head(30).copy()
    top.to_csv(out_dir / "top30_lag_poly.csv", index=False, encoding="utf-8")

    # コンソールにも表示
    print("=== Top lag×poly (observed, filtered, min_count) ===")
    print(top[["hp.ngrc_poly_degree", "hp.ngrc_lag", "count", "median", "mean", "std"]].to_string(index=False))

    # ---- ピボット（ヒートマップ用）----
    pivot_val = summary_ok.pivot(index="hp.ngrc_poly_degree", columns="hp.ngrc_lag", values=args.agg).sort_index()
    pivot_cnt = summary_ok.pivot(index="hp.ngrc_poly_degree", columns="hp.ngrc_lag", values="count").reindex(
        index=pivot_val.index, columns=pivot_val.columns
    )

    # ヒートマップ
    make_heatmap(
        pivot_val,
        pivot_cnt,
        title=f"{args.target} ({args.agg}) by (poly_degree × lag)\nwhere={args.where or 'None'}, min_count={args.min_count}",
        out_path=out_dir / "heatmap_lag_poly.png",
    )

    # ---- 追加：polyごとの曲線（見やすい）----
    fig, ax = plt.subplots(figsize=(10, 6))
    for poly in sorted(summary_ok["hp.ngrc_poly_degree"].unique()):
        sub = summary_ok[summary_ok["hp.ngrc_poly_degree"] == poly].sort_values("hp.ngrc_lag")
        ax.plot(sub["hp.ngrc_lag"], sub[args.agg], marker="o", label=f"poly={int(poly)}")
    ax.set_title(f"{args.target} ({args.agg}) vs lag (grouped by poly)\nwhere={args.where or 'None'}")
    ax.set_xlabel("ngrc_lag")
    ax.set_ylabel(f"{args.target} ({args.agg})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "curve_by_poly.png", dpi=200)
    plt.close(fig)

    # ---- bootstrap CI（任意）----
    if args.bootstrap_ci:
        rows = []
        for (poly, lag), sub in df.groupby(["hp.ngrc_poly_degree", "hp.ngrc_lag"]):
            if len(sub) < args.min_count:
                continue
            vals = sub[args.target].to_numpy(dtype=float)
            lo, hi = bootstrap_ci_median(vals, n_boot=args.n_boot, seed=0)
            rows.append({
                "hp.ngrc_poly_degree": poly,
                "hp.ngrc_lag": lag,
                "n": len(vals),
                "median": float(np.median(vals)),
                "ci_lo": lo,
                "ci_hi": hi,
            })
        ci_df = pd.DataFrame(rows).sort_values("median")
        ci_df.to_csv(out_dir / "lag_poly_median_bootstrap_ci.csv", index=False, encoding="utf-8")

    # ---- d_modelごと（任意）----
    if args.per_d_model and "hp.ngrc_d_model" in df.columns:
        df["hp.ngrc_d_model"] = pd.to_numeric(df["hp.ngrc_d_model"], errors="coerce")
        for d in sorted(df["hp.ngrc_d_model"].dropna().unique()):
            subdf = df[df["hp.ngrc_d_model"] == d].copy()
            if len(subdf) < 5:
                continue
            subdir = out_dir / f"by_d_model_{int(d)}"
            subdir.mkdir(parents=True, exist_ok=True)

            g2 = subdf.groupby(["hp.ngrc_poly_degree", "hp.ngrc_lag"])[args.target]
            s2 = g2.agg(count="size", mean="mean", median="median", std="std").reset_index()
            s2_ok = s2[s2["count"] >= args.min_count].copy()
            s2.to_csv(subdir / "lag_poly_summary_all.csv", index=False, encoding="utf-8")
            s2_ok.to_csv(subdir / "lag_poly_summary_min_count.csv", index=False, encoding="utf-8")

            if len(s2_ok) == 0:
                continue

            pivot_val2 = s2_ok.pivot(index="hp.ngrc_poly_degree", columns="hp.ngrc_lag", values=args.agg).sort_index()
            pivot_cnt2 = s2_ok.pivot(index="hp.ngrc_poly_degree", columns="hp.ngrc_lag", values="count").reindex(
                index=pivot_val2.index, columns=pivot_val2.columns
            )
            make_heatmap(
                pivot_val2,
                pivot_cnt2,
                title=f"{args.target} ({args.agg}) by (poly × lag), d_model={int(d)}\nmin_count={args.min_count}",
                out_path=subdir / "heatmap_lag_poly.png",
            )

            best = s2_ok.sort_values(args.agg).head(10)
            best.to_csv(subdir / "top10_lag_poly.csv", index=False, encoding="utf-8")

    # markdown簡易レポート
    md = []
    md.append(f"# lag×poly optimization report\n")
    md.append(f"- target: `{args.target}` (lower is better)\n")
    md.append(f"- where: `{args.where or 'None'}`\n")
    md.append(f"- min_count: {args.min_count}\n")
    md.append(f"- agg: {args.agg}\n\n")
    md.append("## Top 30 (observed)\n\n")
    md.append(top.to_markdown(index=False))
    md.append("\n\n## Files\n")
    md.append("- heatmap: `heatmap_lag_poly.png`\n")
    md.append("- curves : `curve_by_poly.png`\n")
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\nSaved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
