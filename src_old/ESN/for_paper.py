#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze ESN recurrent-matrix metrics and correlate them with perplexity.
  * input : run_table.csv   (id, seed, final_train_perplexity, final_val_perplexity, model_path)
  * output: run_table_metrics.csv   (metrics columns appended)
           plots/corr_heatmap.png  (pairwise Pearson correlation heat-map)
"""

from pathlib import Path
import argparse, warnings, json, re
from functools import lru_cache
from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.stats import skew, kurtosis
import torch, networkx as nx, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import japanize_matplotlib
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "serif"
# ────────────────  CLI  ────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="run_table_65536_0827.csv",
                help="Input CSV (requires model_path column)")
ap.add_argument("--out_csv", default="run_table_metrics_65536_0827.csv")
ap.add_argument("--plots_dir", default="plots_65536_paper")
ap.add_argument("--power_iters", type=int, default=20000,
                help="Iterations for power method (eigen / singular)")
ap.add_argument("--sample_nodes", type=int, default=2048,
                help="For huge graphs, BFS sample size for avg shortest path")
args = ap.parse_args()

plots_dir = Path(args.plots_dir)
plots_dir.mkdir(exist_ok=True)

df = pd.read_csv(args.out_csv)
metrics_cols = [
    "fro_norm",
    "w_mean", "w_std", "w_sign_ratio", "w_skew", "w_kurt",
    "in_sum_var", "out_sum_var", "sign_balance_var",
]
# ────────────────  CORRELATION HEAT-MAP  ─────────────────────────
corr_cols = metrics_cols + ["final_val_perplexity",]
corr = df[corr_cols].corr(method="pearson")
fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_cols)))
ax.set_yticks(range(len(corr_cols)))
ax.set_xticklabels(corr_cols, rotation=90)
ax.set_yticklabels(corr_cols)
plt.colorbar(im, ax=ax, shrink=0.8)
plt.title("Pearson correlation matrix (metrics ↔ perplexity)")
plt.tight_layout()
fig_path = plots_dir / "corr_heatmap_except_execution.png"
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"📈 correlation heat-map saved → {fig_path}")
# metric名 → LaTeX表記 の対応辞書
latex_labels = {
    "fro_norm": r"$\|W\|_F$",
    "w_mean": r'$\mu_w$',
    "w_std": r'$\sigma_w$',
    "w_sign_ratio": r'$\mathrm{positive\ ratio}$',
    "w_skew": r'$\mathrm{W}_{\mathrm{skewness}}$',
    "w_kurt": r'$\mathrm{W}_{\mathrm{kurtosis}}$',
    "in_sum_var": r'$\mathrm{Var}_{\mathrm{in}}$',
    "out_sum_var": r'$\mathrm{Var}_{\mathrm{out}}$',
    "sign_balance_var": r'$\mathrm{Var}_{\mathrm{SB}}$',
    "final_val_perplexity": r'$\mathrm{PPL}_{\mathrm{val}}^{(\mathrm{final})}$',
}

import matplotlib.ticker as ticker

ppl_col = "final_val_perplexity"

# 論文向けフォントサイズのデフォルト設定
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

fig, axes = plt.subplots(2, 4, figsize=(16, 14))
axes = axes.flatten()

import matplotlib.ticker as ticker

ppl_col = "final_val_perplexity"

# w_sign_ratio を除いた metrics
metrics_cols_for_plot = [m for m in metrics_cols if m != "w_sign_ratio"]

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, metric in enumerate(metrics_cols_for_plot):
    if metric == "fro_norm":
        continue

    ax = axes[i]
    ax.scatter(df[metric], df[ppl_col], alpha=0.6, s=20)


    # x軸は LaTeX 表記
    ax.set_xlabel(latex_labels.get(metric, metric), fontsize=18)

    # y軸は左列だけ表示
    if i % 4 == 0:
        ax.set_ylabel(latex_labels[ppl_col], fontsize=18)
    else:
        ax.set_ylabel("")

    # 軸目盛りの数を減らす
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))

# 余ったsubplotを削除（今回は8個なので不要だけど保険で）
for j in range(len(metrics_cols_for_plot), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
scatter_grid_path = plots_dir / "scatter_metrics_vs_valppl_grid_2x4.png"
plt.savefig(scatter_grid_path, dpi=300, bbox_inches="tight")  # 高解像度
plt.close()
print(f"📉 scatter grid (2x4) saved → {scatter_grid_path}")

plt.tight_layout()
scatter_grid_path = plots_dir / "scatter_metrics_vs_valppl_grid.png"
plt.savefig(scatter_grid_path, dpi=300, bbox_inches="tight")  # 高解像度で保存
plt.close()
print(f"📉 scatter grid saved → {scatter_grid_path}")



# ppl_col = "final_val_perplexity"
# corr_vals = df[metrics_cols + [ppl_col]].corr(method="pearson")[ppl_col].drop(ppl_col)

# # LaTeXラベルに変換
# labels = [latex_labels.get(c, c) for c in corr_vals.index]

# plt.figure(figsize=(10, 6))
# plt.barh(labels, corr_vals.values, color="skyblue",fontsize=17)
# plt.axvline(0, color="k", linestyle="--", linewidth=1)
# plt.xlabel("Pearson correlation", fontsize=17)
# plt.tight_layout()

# bar_path = plots_dir / "metrics_vs_valppl_corr.png"
# plt.savefig(bar_path, dpi=150)
# plt.close()
# print(f"📊 correlation bar plot saved → {bar_path}")