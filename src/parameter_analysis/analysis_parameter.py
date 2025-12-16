#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance


# ----------------------------
# 読み込み（JSONっぽいテキストを頑健に読む）
# ----------------------------

def _strip_code_fences(text: str) -> str:
    # ```json ... ``` みたいな囲いがあっても剥がす
    text = text.strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def load_jsonlike_file(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = _strip_code_fences(text)

    # 先頭/末尾に余計な文字が混ざっても、最初の{〜最後の}を抜く
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError(f"JSON object not found in: {path}")

    blob = text[start : end + 1]

    # 1) まず素直にJSONとして読む
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        # 2) よくある「末尾カンマ」だけ雑に修正して再挑戦
        blob2 = re.sub(r",\s*([}\]])", r"\1", blob)
        return json.loads(blob2)


def flatten_run(obj: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    row["file"] = str(file_path)
    row["run_name"] = obj.get("run_name")

    # 代表的なメタ情報（必要なら増やしてください）
    for k in [
        "parameter_count",
        "max_gpu_memory_MB",
        "training_time_sec",
        "final_train_loss",
        "final_train_perplexity",
        "final_val_loss",
        "final_val_perplexity",
        "seed",
        "inference_tok_per_sec",
        "inference_mem_MB",
    ]:
        if k in obj:
            row[k] = obj.get(k)

    hp = obj.get("hyperparameters") or {}
    if not isinstance(hp, dict):
        hp = {}

    for k, v in hp.items():
        col = f"hp.{k}"
        if isinstance(v, (dict, list)):
            # ネストはJSON文字列に潰す（解析対象にしたいなら別途flatten戦略が必要）
            row[col] = json.dumps(v, ensure_ascii=False, sort_keys=True)
        else:
            row[col] = v

    return row


# ----------------------------
# 前処理（数値/カテゴリ判定、定数列除去など）
# ----------------------------

def infer_column_types(X: pd.DataFrame) -> Tuple[List[str], List[str], pd.DataFrame]:
    """
    - できるだけ数値に変換できる列は numeric
    - それ以外は categorical
    """
    X = X.copy()
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in X.columns:
        s = X[col]

        # boolはカテゴリ扱い（True/Falseの選択として見る）
        if pd.api.types.is_bool_dtype(s):
            categorical_cols.append(col)
            continue

        # 数値化を試す（9割以上変換成功なら数値扱い）
        conv = pd.to_numeric(s, errors="coerce")
        ok_ratio = conv.notna().mean()
        if ok_ratio >= 0.90:
            X[col] = conv
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols, X


def drop_constant_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
    kept = []
    for c in cols:
        nunique = df[c].nunique(dropna=True)
        if nunique >= 2:
            kept.append(c)
    return kept


# ----------------------------
# 重要度推定（Permutation Importance + CV）
# ----------------------------

@dataclass
class ImportanceResult:
    importances_mean: pd.Series
    importances_std: pd.Series
    cv_mae: float
    cv_mae_std: float


def compute_permutation_importance_cv(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: List[str],
    categorical_cols: List[str],
    random_state: int = 42,
) -> ImportanceResult:
    # 前処理
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=600,
        random_state=random_state,
        min_samples_leaf=2,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[("prep", preprocess), ("model", model)])

    n = len(X)
    # 小規模でも破綻しないよう分割数を調整（50件なら5分割）
    n_splits = min(5, max(2, n // 10))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    imp_list = []
    mae_list = []

    for tr_idx, te_idx in cv.split(X):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        pipe.fit(X_tr, y_tr)

        # CV評価（MAE）
        pred = pipe.predict(X_te)
        mae = float(np.mean(np.abs(pred - y_te.to_numpy())))
        mae_list.append(mae)

        # permutation importance（テスト側で測る）
        r = permutation_importance(
            pipe,
            X_te,
            y_te,
            n_repeats=40,
            random_state=random_state,
            scoring="neg_mean_absolute_error",
        )
        imp_list.append(pd.Series(r.importances_mean, index=X.columns))

    imp_mat = pd.concat(imp_list, axis=1)
    imp_mean = imp_mat.mean(axis=1).sort_values(ascending=False)
    imp_std = imp_mat.std(axis=1).reindex(imp_mean.index)

    mae_arr = np.array(mae_list, dtype=float)
    return ImportanceResult(
        importances_mean=imp_mean,
        importances_std=imp_std,
        cv_mae=float(mae_arr.mean()),
        cv_mae_std=float(mae_arr.std(ddof=1)) if len(mae_arr) >= 2 else 0.0,
    )


# ----------------------------
# 可視化
# ----------------------------

def plot_importance_bar(imp: pd.Series, out_path: Path, topk: int = 20) -> None:
    top = imp.head(topk)[::-1]  # barh用に反転
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(top))))
    ax.barh(top.index, top.values)
    ax.set_title(f"Permutation Importance (Top {topk})")
    ax.set_xlabel("Importance (MAE increase; larger = more important)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_scatter_top_numeric(
    df: pd.DataFrame,
    target: str,
    numeric_params: List[str],
    out_path: Path,
    max_plots: int = 9,
) -> None:
    # まとめて保存（縦に並べる）
    chosen = numeric_params[:max_plots]
    if len(chosen) == 0:
        return

    fig, axes = plt.subplots(len(chosen), 1, figsize=(10, 3.2 * len(chosen)), sharey=True)
    if len(chosen) == 1:
        axes = [axes]

    y = df[target].to_numpy()

    for ax, col in zip(axes, chosen):
        x = df[col].to_numpy()
        ax.scatter(x, y, s=18)
        ax.set_title(f"{col} vs {target}")
        ax.set_xlabel(col)
        ax.set_ylabel(target)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_boxplot_top_categorical(
    df: pd.DataFrame,
    target: str,
    cat_params: List[str],
    out_path: Path,
    max_plots: int = 6,
) -> None:
    chosen = cat_params[:max_plots]
    if len(chosen) == 0:
        return

    fig, axes = plt.subplots(len(chosen), 1, figsize=(12, 3.2 * len(chosen)), sharey=True)
    if len(chosen) == 1:
        axes = [axes]

    for ax, col in zip(axes, chosen):
        sub = df[[col, target]].dropna()
        # カテゴリ数が多すぎると破綻するので上位だけに制限
        counts = sub[col].value_counts()
        keep = counts.head(12).index
        sub = sub[sub[col].isin(keep)]

        groups = []
        labels = []
        for k in keep:
            vals = sub[sub[col] == k][target].to_numpy()
            if len(vals) >= 1:
                groups.append(vals)
                labels.append(str(k))

        if len(groups) == 0:
            continue

        ax.boxplot(groups, labels=labels, vert=True)
        ax.set_title(f"{col} (top categories) vs {target}")
        ax.set_xlabel(col)
        ax.set_ylabel(target)
        ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ----------------------------
# 方向感（Spearman相関）とカテゴリ平均
# ----------------------------

def spearman_for_numeric(df: pd.DataFrame, numeric_cols: List[str], target: str) -> pd.DataFrame:
    rows = []
    y = df[target]
    for c in numeric_cols:
        x = df[c]
        sub = pd.concat([x, y], axis=1).dropna()
        if len(sub) < 5:
            continue
        rho, p = spearmanr(sub[c].to_numpy(), sub[target].to_numpy())
        rows.append({
            "param": c,
            "spearman_rho": float(rho),
            "p_value": float(p),
            "n": int(len(sub)),
            "min": float(sub[c].min()),
            "max": float(sub[c].max()),
        })
    out = pd.DataFrame(rows).sort_values("spearman_rho")
    return out


def categorical_means(df: pd.DataFrame, cat_cols: List[str], target: str) -> pd.DataFrame:
    rows = []
    for c in cat_cols:
        sub = df[[c, target]].dropna()
        if len(sub) < 5:
            continue
        vc = sub[c].value_counts(dropna=True)
        # カテゴリが細かすぎる列は扱いにくいので、ある程度に制限
        if vc.shape[0] > 30:
            continue
        for cat, n in vc.items():
            vals = sub[sub[c] == cat][target].to_numpy()
            rows.append({
                "param": c,
                "category": str(cat),
                "n": int(n),
                "mean_target": float(np.mean(vals)),
                "median_target": float(np.median(vals)),
            })
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    # 各param内で平均が小さい順（=良い）に並ぶように
    out = out.sort_values(["param", "mean_target"], ascending=[True, True])
    return out


# ----------------------------
# メイン
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True, help="実験ログ(JSON風テキスト)のあるディレクトリ")
    ap.add_argument("--pattern", type=str, default="*.txt", help="読み込むファイルのglobパターン（例: *.txt, *.json）")
    ap.add_argument("--target", type=str, default="final_train_perplexity", help="目的変数（小さいほど良い想定）")
    ap.add_argument("--out_dir", type=str, default="./analysis_out_NGRC", help="出力先ディレクトリ")
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(args.pattern))
    if len(files) == 0:
        raise SystemExit(f"No files matched: {in_dir} / {args.pattern}")

    rows = []
    errors = []
    for fp in files:
        try:
            obj = load_jsonlike_file(fp)
            row = flatten_run(obj, fp)
            rows.append(row)
        except Exception as e:
            errors.append((str(fp), str(e)))

    df = pd.DataFrame(rows)

    # 目的変数がない行は落とす
    if args.target not in df.columns:
        raise SystemExit(f"Target '{args.target}' not found in parsed columns.\nColumns: {list(df.columns)[:50]} ...")

    df = df.dropna(subset=[args.target]).reset_index(drop=True)

    # 解析対象：hp.* だけ
    hp_cols = [c for c in df.columns if c.startswith("hp.")]
    if len(hp_cols) == 0:
        raise SystemExit("No hyperparameter columns found (hp.*).")

    X = df[hp_cols].copy()
    y = df[args.target].copy()

    # 型推定 + 定数列除去
    num_cols, cat_cols, X_typed = infer_column_types(X)
    num_cols = drop_constant_columns(X_typed, num_cols)
    cat_cols = drop_constant_columns(X_typed, cat_cols)

    X_typed = X_typed[num_cols + cat_cols].copy()

    # 保存（パース結果）
    df_out = pd.concat([df[["file", "run_name"]], X_typed, df[[args.target]]], axis=1)
    df_out.to_csv(out_dir / "runs_parsed.csv", index=False, encoding="utf-8")

    # 重要度（CV permutation）
    imp_res = compute_permutation_importance_cv(
        X=X_typed,
        y=y,
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        random_state=args.random_state,
    )

    imp_df = pd.DataFrame({
        "param": imp_res.importances_mean.index,
        "importance_mean": imp_res.importances_mean.values,
        "importance_std": imp_res.importances_std.values,
    }).sort_values("importance_mean", ascending=False)
    imp_df.to_csv(out_dir / "importance_permutation_cv.csv", index=False, encoding="utf-8")

    # 相関（数値のみ）
    sp = spearman_for_numeric(pd.concat([X_typed, y.rename(args.target)], axis=1), num_cols, args.target)
    sp.to_csv(out_dir / "spearman_numeric.csv", index=False, encoding="utf-8")

    # カテゴリ平均
    catm = categorical_means(pd.concat([X_typed, y.rename(args.target)], axis=1), cat_cols, args.target)
    catm.to_csv(out_dir / "categorical_means.csv", index=False, encoding="utf-8")

    # 可視化
    plot_importance_bar(imp_res.importances_mean, out_dir / "importance_top20.png", topk=20)

    # 重要度上位から「数値」「カテゴリ」それぞれ上位を抽出してプロット
    top_params = imp_res.importances_mean.index.tolist()
    top_numeric = [p for p in top_params if p in num_cols]
    top_cat = [p for p in top_params if p in cat_cols]

    df_plot = pd.concat([X_typed, y.rename(args.target)], axis=1)

    plot_scatter_top_numeric(df_plot, args.target, top_numeric, out_dir / "scatter_top_numeric.png", max_plots=9)
    plot_boxplot_top_categorical(df_plot, args.target, top_cat, out_dir / "boxplot_top_categorical.png", max_plots=6)

    # レポート（Markdown）
    best_idx = int(df[args.target].idxmin())
    worst_idx = int(df[args.target].idxmax())
    best_row = df.loc[best_idx]
    worst_row = df.loc[worst_idx]

    lines = []
    lines.append(f"# Hyperparameter Contribution Report\n")
    lines.append(f"- Files parsed: {len(files)}")
    lines.append(f"- Rows used (target not-null): {len(df)}")
    if errors:
        lines.append(f"- Parse errors: {len(errors)} (see below)\n")
    lines.append("")
    lines.append(f"## Target\n")
    lines.append(f"- target: `{args.target}`  (lower is better)\n")
    lines.append(f"## CV sanity check\n")
    lines.append(f"- CV MAE (mean ± std): {imp_res.cv_mae:.6g} ± {imp_res.cv_mae_std:.6g}\n")

    lines.append("## Best / Worst run\n")
    lines.append(f"- Best: `{best_row.get('run_name')}`  ({args.target}={best_row.get(args.target)})")
    lines.append(f"- Worst: `{worst_row.get('run_name')}` ({args.target}={worst_row.get(args.target)})\n")

    lines.append("## Top 15 important hyperparameters (Permutation Importance)\n")
    lines.append(imp_df.head(15).to_markdown(index=False))
    lines.append("")

    if len(sp) > 0:
        lines.append("## Numeric params: Spearman correlation hint (direction)\n")
        lines.append("- rho < 0 : 値が大きいほど target が下がりやすい（改善しやすい傾向）")
        lines.append("- rho > 0 : 値が大きいほど target が上がりやすい（悪化しやすい傾向）\n")
        lines.append(sp.sort_values("p_value").head(20).to_markdown(index=False))
        lines.append("")

    if len(catm) > 0:
        lines.append("## Categorical params: category mean (lower is better)\n")
        lines.append("（各param内で平均が小さいカテゴリが良い傾向）\n")
        # 各paramの上位1カテゴリだけ抜粋
        best_cats = catm.sort_values(["param", "mean_target"]).groupby("param", as_index=False).head(1)
        lines.append(best_cats.to_markdown(index=False))
        lines.append("")

    if errors:
        lines.append("## Parse errors\n")
        for fp, msg in errors[:50]:
            lines.append(f"- {fp}: {msg}")
        if len(errors) > 50:
            lines.append(f"- ... and {len(errors)-50} more")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("Done.")
    print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
