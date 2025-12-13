# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# Analyze ESN recurrent-matrix metrics and correlate them with perplexity.
#   * input : run_table.csv   (id, seed, final_train_perplexity, final_val_perplexity, model_path)
#   * output: run_table_metrics.csv   (metrics columns appended)
#            plots/corr_heatmap.png  (pairwise Pearson correlation heat-map)
# """

# from pathlib import Path
# import argparse, warnings, json, re
# from functools import lru_cache
# from tqdm import tqdm
# import numpy as np
# import pandas as pd
# import scipy.sparse as sp
# import scipy.sparse.linalg as spla
# from scipy.stats import skew, kurtosis
# import torch, networkx as nx, matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler

# # ────────────────  CLI  ────────────────────────────────────────────
# ap = argparse.ArgumentParser()
# ap.add_argument("--csv", default="run_table_65536_0827.csv",
#                 help="Input CSV (requires model_path column)")
# ap.add_argument("--out_csv", default="run_table_metrics_65536_0827.csv")
# ap.add_argument("--plots_dir", default="plots_65536")
# ap.add_argument("--power_iters", type=int, default=20000,
#                 help="Iterations for power method (eigen / singular)")
# ap.add_argument("--sample_nodes", type=int, default=2048,
#                 help="For huge graphs, BFS sample size for avg shortest path")
# ap.add_argument("--ignore_perplexity_limit", default=True, help="Ignore perplexity limit")
# args = ap.parse_args()

# plots_dir = Path(args.plots_dir)
# plots_dir.mkdir(exist_ok=True)

# # ────────────────  UTILS  ──────────────────────────────────────────

# def load_latest_pt(path: Path) -> Path:
#     """path が .pt ファイルならそのまま、フォルダなら再帰で最新 .pt を返す"""
#     print(path)
#     if path.is_file() and path.suffix == ".pt":
#         return path
#     if path.is_dir():
#         pts = sorted(path.rglob("*.pt"),        # 再帰探索
#                      key=lambda p: p.stat().st_mtime,
#                      reverse=True)
#         if pts:
#             return pts[0]
#     raise FileNotFoundError(f"No .pt file found under {path}")
# @lru_cache(maxsize=32)
# def fetch_mat_a(ckpt_path: str):
#     """Return scipy.sparse.coo_matrix (float64) & leak-rate numpy array (float64)"""
#     sd = torch.load(ckpt_path, map_location="cpu")

#     # ---- W_rec を探す（直下 or model/module/state_dict 配下） ----
#     mat = None
#     for k in ("W_rec", "w_rec", "W_rec_coo"):
#         if k in sd:
#             mat = sd[k]; break
#     if mat is None:
#         for top in ("model", "module", "state_dict"):
#             if top in sd:
#                 sub = sd[top]
#                 for k in ("W_rec", "w_rec", "W_rec_coo"):
#                     if k in sub:
#                         mat = sub[k]; break
#             if mat is not None:
#                 break
#     if mat is None:
#         raise KeyError("W_rec not found in checkpoint")

#     # ---- COO にまとめて dtype を float32/float64 に統一（bfloat16 対策） ----
#     if not isinstance(mat, torch.Tensor):
#         raise TypeError("W_rec is not a torch.Tensor")
#     mat = mat.coalesce().to(dtype=torch.float32, device="cpu")   # ← ここが重要

#     idx = mat.indices().cpu().numpy()                            # shape (2, nnz), int64
#     data = mat.values().to(torch.float64).cpu().numpy()          # ← float64 に上げる
#     shape = tuple(mat.size())

#     # ---- scipy COO に変換 ----
#     W = sp.coo_matrix((data, (idx[0], idx[1])), shape=shape)

#     # ---- leak rate a ----
#     a = None
#     for k in ("a",):
#         if k in sd: a = sd[k]
#     if a is None:
#         for top in ("model", "module", "state_dict"):
#             if top in sd and "a" in sd[top]:
#                 a = sd[top]["a"]
#                 break
#     if a is not None:
#         a = a.to(dtype=torch.float64, device="cpu").numpy()

#     return W, a

# import numpy as np
# import scipy.sparse as sp

# def power_eig_radius(
#     A: sp.csr_matrix,
#     max_iter: int = 200,
#     tol: float = 1e-6,
#     second: bool = False,
# ):
#     """
#     Parameters
#     ----------
#     A : sp.csr_matrix
#         対象行列（疎・正方）
#     max_iter : int
#         最大イテレーション数 (fallback の上限)
#     tol : float
#         ρ の相対更新量が `tol` 未満になったら収束とみなす
#     second : bool
#         True のとき第2固有値 |λ₂| も deflation で推定する

#     Returns
#     -------
#     rho1 : float
#         最大固有値絶対値 (= スペクトル半径)
#     rho2 : float | None
#         |λ₂|（second=True のときのみ）
#     iters1 : int
#         ρ₁ の収束に要したイテレーション数
#     iters2 : int | None
#         ρ₂ の収束イテレーション（second=True のとき）
#     """
#     n = A.shape[0]

#     # ── λ₁ ─────────────────────────────────────────────
#     v = np.random.randn(n, 1)
#     v /= np.linalg.norm(v)
#     rho_old = 0.0
#     with tqdm(total=max_iter, desc="power iteration") as pbar:
#         for k in range(1, max_iter + 1):
#             pbar.update(1)
#             v = A @ v
#             v /= np.linalg.norm(v)
#             rho = float(abs((A @ v).T @ v)[0, 0])
#             if abs(rho - rho_old) / (rho_old + 1e-12) < tol:
#                 break
#             rho_old = rho
#     rho1, iters1 = rho, k

#     if not second:
#         return rho1, None

#     # ── λ₂ (deflation) ────────────────────────────────
#     Av   = A @ v
#     Adef = A - (Av @ v.T)            # rank-1 deflation

#     w = np.random.randn(n, 1)
#     w -= v @ (v.T @ w)               # v と直交に
#     w /= np.linalg.norm(w)
#     rho_old = 0.0
#     with tqdm(total=max_iter, desc="power iteration") as pbar:
#         for k in range(1, max_iter + 1):
#             pbar.update(1)
#             w = Adef @ w
#             # v 方向成分を都度引いて直交性維持
#             w -= v @ (v.T @ w)
#             w /= np.linalg.norm(w)
#             rho = float(abs((A @ w).T @ w)[0, 0])
#             if abs(rho - rho_old) / (rho_old + 1e-12) < tol:
#                 break
#             rho_old = rho
#     rho2, iters2 = rho, k

#     return rho1, rho2, iters1, iters2
# # def power_eig_radius(A: sp.csr_matrix, iters=200, second=False):
# #     """return ρ_max (and optionally ρ_2) via power iteration / deflation"""
# #     n = A.shape[0]
# #     v = np.random.randn(n, 1)
# #     v /= np.linalg.norm(v)
# #     for _ in range(iters):
# #         v = A @ v
# #         v /= np.linalg.norm(v)
# #     lam1 = float(np.abs((A @ v).T @ v))
# #     if not second:
# #         return lam1, None
# #     # deflation for λ₂ magnitude
# #     Av = (A @ v)
# #     A_def = A - (Av @ v.T)  # rank-1 deflation
# #     w = np.random.randn(n, 1)
# #     w -= v @ (v.T @ w)
# #     w /= np.linalg.norm(w)
# #     for _ in range(iters):
# #         w = A_def @ w
# #         w -= v @ (v.T @ w)  # keep orthogonal
# #         w /= np.linalg.norm(w)
# #     lam2 = float(np.abs((A @ w).T @ w))
# #     return lam1, lam2

# def largest_smallest_sv(A: sp.csr_matrix, iters=200):
#     """largest σ_max and smallest σ_min using svds (k=1) twice"""
#     σ_max = spla.svds(A, k=1, return_singular_vectors=False)[0]
#     σ_min = spla.svds(A, k=1, return_singular_vectors=False, which="SM")[0]
#     return float(σ_max), float(σ_min)

# def weight_stats(data):
#     mean = float(data.mean())
#     std  = float(data.std())
#     sign_ratio = float((data > 0).mean())          # fraction positive
#     skewness   = float(skew(data))
#     kurt       = float(kurtosis(data))
#     return mean, std, sign_ratio, skewness, kurt

# def self_loops(idx):
#     return int(np.sum(idx[0] == idx[1]))

# # def graph_metrics(idx, n_nodes, sample_nodes=2048):
# #     """Return SCC stats, avg shortest path, diameter, degree stats"""
# #     G = nx.DiGraph()
# #     G.add_nodes_from(range(n_nodes))
# #     G.add_edges_from(zip(idx[0], idx[1]))

# #     # SCC sizes
# #     sccs = [len(c) for c in nx.strongly_connected_components(G)]
# #     max_scc = max(sccs)
# #     n_scc   = len(sccs)

# #     # undirected largest component for path stats
# #     UG = G.to_undirected()
# #     largest_cc = max(nx.connected_components(UG), key=len)
# #     H = UG.subgraph(largest_cc).copy()
# #     # sample nodes if too large
# #     if len(H) > sample_nodes:
# #         import random
# #         sample = random.sample(list(H.nodes), sample_nodes)
# #         H = H.subgraph(sample).copy()

# #     try:
# #         avg_sp = nx.average_shortest_path_length(H)
# #         diam   = nx.diameter(H)
# #     except nx.NetworkXError:
# #         avg_sp = np.nan
# #         diam   = np.nan

# #     degs = np.array([d for _, d in H.degree()])
# #     deg_skew = float(skew(degs))
# #     deg_var  = float(degs.var())

# #     return max_scc, n_scc, avg_sp, diam, deg_skew, deg_var
# import numpy as np
# import networkx as nx
# import random
# from scipy.stats import skew

# def graph_metrics(idx, n_nodes, sample_nodes=2048):
#     """
#     idx: 2×nnz の ndarray ([rows, cols])
#     n_nodes: 全ノード数 N
#     sample_nodes: サブサンプル時のノード数上限
#     returns:
#       scc_max, scc_count,
#       avg_shortest_path, diameter,
#       deg_skew, deg_var
#     """
#     # 有向グラフを作成
#     G = nx.DiGraph()
#     G.add_nodes_from(range(n_nodes))
#     if idx.size > 0:
#         G.add_edges_from(zip(idx[0], idx[1]))

#     # ── 強連結成分 (SCC) ─────────────────────────────
#     sccs = list(nx.strongly_connected_components(G))
#     scc_sizes = [len(c) for c in sccs]
#     scc_max   = max(scc_sizes) if scc_sizes else 0
#     scc_count = len(scc_sizes)

#     # ── 無向化して最大連結成分を取る ────────────────────
#     UG = G.to_undirected()
#     comps = list(nx.connected_components(UG))
#     if not comps:
#         # ノードはいるがエッジが一切ない場合
#         avg_sp, diam = np.nan, np.nan
#         deg_skew, deg_var = np.nan, np.nan
#     else:
#         # 最大成分
#         largest = max(comps, key=len)
#         H = UG.subgraph(largest).copy()
#         # ノード数が多すぎるときはサンプリング
#         if H.number_of_nodes() > sample_nodes:
#             sample = random.sample(list(H.nodes), sample_nodes)
#             H = UG.subgraph(sample).copy()

#         # 平均最短経路 & 直径
#         if H.number_of_nodes() <= 1:
#             avg_sp, diam = 0.0, 0.0
#         else:
#             # 念のため再チェック
#             if nx.is_connected(H):
#                 avg_sp = nx.average_shortest_path_length(H)
#                 diam   = nx.diameter(H)
#             else:
#                 avg_sp, diam = np.nan, np.nan

#         # 次数分布統計
#         degs = np.fromiter((d for _, d in H.degree()), dtype=float)
#         deg_skew = skew(degs) if degs.size>0 else np.nan
#         deg_var  = degs.var() if degs.size>0 else np.nan

#     return scc_max, scc_count, avg_sp, diam, deg_skew, deg_var
# def moment_trace(A: sp.csr_matrix, k=2, n_probe=32):
#     """Hutch++: approximate trace(A^k) / n"""
#     n = A.shape[0]
#     def Av(v): return A @ v
#     v = np.random.randn(n, n_probe)
#     res = 0.0
#     for i in range(n_probe):
#         x = v[:, i:i+1]
#         for _ in range(k):
#             x = Av(x)
#         res += float((v[:, i:i+1].T @ x))
#     return res / n_probe / n    # normalize by n

# def node_sums(idx, data, n):
#     """per-node |in| & |out| sums, sign-balance"""
#     src, dst = idx
#     abs_vals = np.abs(data)
#     in_sum  = np.bincount(dst, weights=abs_vals, minlength=n)
#     out_sum = np.bincount(src, weights=abs_vals, minlength=n)
#     pos_vals = data > 0
#     pos_in  = np.bincount(dst, weights=pos_vals, minlength=n)
#     neg_in  = np.bincount(dst, weights=~pos_vals, minlength=n)
#     # sign balance   (pos - neg) / (pos + neg + ε)
#     ε = 1e-9
#     sb = (pos_in - neg_in) / (pos_in + neg_in + ε)
#     return (float(in_sum.var()),
#             float(out_sum.var()),
#             float(sb.var()))

# # ────────────────  MAIN LOOP  ──────────────────────────────────────
# df = pd.read_csv(args.csv)
# metrics_cols = [
#     #"rho_act", "eig_gap", "sigma_max", "sigma_min", "cond",
#     #"rho_act",
#     "fro_norm",
#     "w_mean", "w_std", "w_sign_ratio", "w_skew", "w_kurt",
#     #"scc_max", "scc_count",
#     #"avg_shortest_path", " diameter",
#     # "deg_skew", "deg_var",
#     #"moment2_trace", "moment3_trace",
#     "in_sum_var", "out_sum_var", "sign_balance_var",
# ]
# for c in metrics_cols:
#     df[c] = np.nan

# for i, row in tqdm(df.iterrows(), total=len(df), desc="computing metrics"):
#     try:
#         ckpt = load_latest_pt(Path(row["model_path"]))
#         print(ckpt)
#         W, a = fetch_mat_a(str(ckpt))
#     except Exception as e:
#         print(e)
#         continue
#     # if row["final_val_perplexity"] > 1000:
#     #     continue
#     #if row["val_ppl"]
#     n = W.shape[0]
#     idx = np.vstack((W.row, W.col))
#     data = W.data
#     fro_norm = np.linalg.norm(data)
#     # ---------- eigen / singular ----------
#     print("eigen / singular")
#     #rho, rho2 = power_eig_radius(W.tocsr(), args.power_iters,tol=1e-6, second=False)
#     # σ_max, σ_min = largest_smallest_sv(W.tocsr())
#     # cond = σ_max / (σ_min + 1e-12)

#     #σ_max, σ_min = None, None
#     #cond = None
    

#     # ---------- weight stats ----------
#     print("weight stats")
#     w_mean, w_std, w_sign, w_sk, w_kurt = weight_stats(data)

#     # ---------- graph ----------
#     # print("graph")
#     # sl = self_loops(idx)
#     # scc_max, scc_cnt, avg_sp, diam, deg_sk, deg_var = graph_metrics(
#     #     idx, n, args.sample_nodes
#     # )
#     # print(scc_max, scc_cnt, avg_sp, diam, deg_sk, deg_var)

#     # ---------- moments ----------
#     # print("moments")
#     # m2 = moment_trace(W.tocsr(), k=2)
#     # m3 = moment_trace(W.tocsr(), k=3)

#     # ---------- node sums ----------
#     print("node sums")
#     in_var, out_var, sb_var = node_sums(idx, data, n)

#     # ---------- assign ----------
#     df.loc[i, metrics_cols] = [
#         #rho, (rho - rho2) if rho2 else np.nan, σ_max, σ_min, cond,
#         #rho,
#         np.linalg.norm(data),
#         w_mean, w_std, w_sign, w_sk, w_kurt,
#         #sl, #scc_max, scc_cnt,
#         #avg_sp, diam,
#         #deg_sk, deg_var,
#         #m2, m3,
#         in_var, out_var, sb_var,
#     ]        

# # ────────────────  SAVE  ──────────────────────────────────────────
# df.to_csv(args.out_csv, index=False)
# print(f"✅ metrics appended → {args.out_csv}")
# ## args.ignore_perplexity_limit が True の場合、validation perplexity の制限を無視
# if args.ignore_perplexity_limit:
#     df = df[df["final_val_perplexity"] < 900]
#     df.to_csv(args.out_csv, index=False)
#     print(f"✅ metrics appended (ignore perplexity limit) → {args.out_csv}")

# # ────────────────  CORRELATION HEAT-MAP  ─────────────────────────
# corr_cols = metrics_cols + ["final_train_perplexity", "final_val_perplexity", "msf_ppl"]
# corr = df[corr_cols].corr(method="pearson")
# fig, ax = plt.subplots(figsize=(14, 12))
# im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
# ax.set_xticks(range(len(corr_cols)))
# ax.set_yticks(range(len(corr_cols)))
# ax.set_xticklabels(corr_cols, rotation=90)
# ax.set_yticklabels(corr_cols)
# plt.colorbar(im, ax=ax, shrink=0.8)
# plt.title("Pearson correlation matrix (metrics ↔ perplexity)")
# plt.tight_layout()
# fig_path = plots_dir / "corr_heatmap_except_execution.png"
# plt.savefig(fig_path, dpi=150)
# plt.close()
# print(f"📈 correlation heat-map saved → {fig_path}")

# # ────────────────  各指標 vs ppl の散布図を作成・保存 ─────────────────────────
# for metric in metrics_cols:
#     for ppl_col in ["final_train_perplexity", "final_val_perplexity", "msf_ppl"]:
#         plt.figure()
#         plt.scatter(df[metric], df[ppl_col], alpha=0.6)
#         plt.xlabel(metric)
#         plt.ylabel(ppl_col)
#         plt.title(f"{metric} vs {ppl_col}")
#         plt.tight_layout()
#         scatter_path = plots_dir / f"{metric}_vs_{ppl_col}.png"
#         plt.savefig(scatter_path, dpi=150)
#         plt.close()
#         print(f"📊 散布図を保存 → {scatter_path}")


# # ────────────────  Box Plot (weights vs perplexity) ─────────────────────────
# # どの perplexity を使うか（例: validation）
# ppl_col = "final_val_perplexity"

# # 有効な行だけ抽出し、perplexity 昇順に並べ替え
# df_sorted = df.dropna(subset=[ppl_col]).sort_values(by=ppl_col)

# # 各 run ごとに W_rec.data を取り出してリストに格納
# all_weights = []
# x_labels = []
# for i, row in tqdm(df_sorted.iterrows(), total=len(df_sorted), desc="collect weights for boxplot"):
#     try:
#         ckpt = load_latest_pt(Path(row["model_path"]))
#         W, a = fetch_mat_a(str(ckpt))
#         data = W.data
#         all_weights.append(data)
#         x_labels.append(f"{row[ppl_col]:.2f}")  # ラベルは ppl 値（小数2桁）
#     except Exception as e:
#         print(f"skip {row['model_path']}: {e}")
#         continue

# # 箱ひげ図を描画
# plt.figure(figsize=(max(10, len(all_weights)*0.5), 6))
# plt.boxplot(all_weights, positions=range(len(all_weights)), showfliers=False)
# plt.xticks(range(len(all_weights)), x_labels, rotation=90)
# plt.xlabel(ppl_col)
# plt.ylabel("Weight values")
# plt.title(f"Weight distributions (box plot) sorted by {ppl_col}")
# plt.tight_layout()

# boxplot_path = plots_dir / f"weights_boxplot_sorted_by_{ppl_col}.png"
# plt.savefig(boxplot_path, dpi=150)
# plt.close()
# print(f"📦 box plot を保存 → {boxplot_path}")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze ESN reservoir matrix metrics and relate them to perplexity, with
degree-normalized indicators, robust correlations, and partial correlations.

Inputs
------
CSV with columns at least:
  id, seed, final_train_perplexity, final_val_perplexity, model_path
Optionally:
  msf_ppl

Outputs
-------
- <out_csv>: metrics appended to the input table
- <plots_dir>/corr_heatmap_*.png : heatmaps (pearson/spearman/partial)
- <plots_dir>/scatter_<metric>_vs_<ppl>.png : scatter plots
- <plots_dir>/corr_tables/*.csv : correlation tables as CSV
- <plots_dir>/weights_boxplot_sorted_by_<ppl>.png : weight distributions
"""

from pathlib import Path
import argparse
from functools import lru_cache
from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.stats import skew, kurtosis, spearmanr, mstats
import torch, networkx as nx, matplotlib.pyplot as plt

# ────────────────  CLI  ────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="run_table_65536_0827.csv",
                help="Input CSV (requires model_path column)")
ap.add_argument("--out_csv", default="run_table_metrics_65536_0827.csv")
ap.add_argument("--plots_dir", default="plots_65536")
ap.add_argument("--power_iters", type=int, default=20000,
                help="Iterations for power method (eigen / singular)")
ap.add_argument("--sample_nodes", type=int, default=2048,
                help="For huge graphs, node sample size (avg shortest path, etc.)")
# Filtering
ap.add_argument("--val_ppl_max", type=float, default=None,
                help="If set, keep rows with final_val_perplexity <= this")
ap.add_argument("--train_ppl_max", type=float, default=None,
                help="If set, keep rows with final_train_perplexity <= this")
# Viz / stats options
ap.add_argument("--log_ppl", action="store_true",
                help="Use log10(PPL) in correlation/plots")
ap.add_argument("--winsor_q", type=float, default=0.0,
                help="Winsorize metrics and PPL by this quantile (e.g., 0.01)")
ap.add_argument("--make_boxplot", default=True,
                help="Collect per-run weight arrays and draw boxplot vs PPL")
ap.add_argument("--make_boxplot_sums", default=True,
                help="Also draw boxplots for node-level input/output sums")
ap.add_argument("--box_yscale", default="linear", choices=["linear", "log"],
                help="y-axis scale for boxplots (linear or log)")
args = ap.parse_args()

plots_dir = Path(args.plots_dir)
plots_dir.mkdir(exist_ok=True)
(plots_dir / "corr_tables").mkdir(exist_ok=True)

# ────────────────  UTILS  ──────────────────────────────────────────
import numpy as np
import scipy.sparse as sp

def power_eig_radius(
    A: sp.csr_matrix,
    max_iter: int = 200,
    tol: float = 1e-6,
    second: bool = False,
):
    """
    Parameters
    ----------
    A : sp.csr_matrix
        対象行列（疎・正方）
    max_iter : int
        最大イテレーション数 (fallback の上限)
    tol : float
        ρ の相対更新量が `tol` 未満になったら収束とみなす
    second : bool
        True のとき第2固有値 |λ₂| も deflation で推定する

    Returns
    -------
    rho1 : float
        最大固有値絶対値 (= スペクトル半径)
    rho2 : float | None
        |λ₂|（second=True のときのみ）
    iters1 : int
        ρ₁ の収束に要したイテレーション数
    iters2 : int | None
        ρ₂ の収束イテレーション（second=True のとき）
    """
    n = A.shape[0]

    # ── λ₁ ─────────────────────────────────────────────
    v = np.random.randn(n, 1)
    v /= np.linalg.norm(v)
    rho_old = 0.0
    with tqdm(total=max_iter, desc="power iteration") as pbar:
        for k in range(1, max_iter + 1):
            pbar.update(1)
            v = A @ v
            v /= np.linalg.norm(v)
            rho = float(abs((A @ v).T @ v)[0, 0])
            if abs(rho - rho_old) / (rho_old + 1e-12) < tol:
                break
            rho_old = rho
    rho1, iters1 = rho, k

    if not second:
        return rho1, None

    # ── λ₂ (deflation) ────────────────────────────────
    Av   = A @ v
    Adef = A - (Av @ v.T)            # rank-1 deflation

    w = np.random.randn(n, 1)
    w -= v @ (v.T @ w)               # v と直交に
    w /= np.linalg.norm(w)
    rho_old = 0.0
    with tqdm(total=max_iter, desc="power iteration") as pbar:
        for k in range(1, max_iter + 1):
            pbar.update(1)
            w = Adef @ w
            # v 方向成分を都度引いて直交性維持
            w -= v @ (v.T @ w)
            w /= np.linalg.norm(w)
            rho = float(abs((A @ w).T @ w)[0, 0])
            if abs(rho - rho_old) / (rho_old + 1e-12) < tol:
                break
            rho_old = rho
    rho2, iters2 = rho, k

    return rho1, rho2, iters1, iters2
def load_latest_pt(path: Path) -> Path:
    """path が .pt ファイルならそのまま、フォルダなら再帰で最新 .pt を返す"""
    if path.is_file() and path.suffix == ".pt":
        return path
    if path.is_dir():
        pts = sorted(path.rglob("*.pt"),
                     key=lambda p: p.stat().st_mtime,
                     reverse=True)
        if pts:
            return pts[0]
    raise FileNotFoundError(f"No .pt file found under {path}")

@lru_cache(maxsize=64)
def fetch_mat_a(ckpt_path: str):
    """Return scipy.sparse.coo_matrix (float64) & leak-rate numpy array (float64 or None)"""
    sd = torch.load(ckpt_path, map_location="cpu")

    # ---- W_rec を探索 ----
    keys = ("W_rec", "w_rec", "W_rec_coo")
    mat = None
    for k in keys:
        if k in sd:
            mat = sd[k]; break
    if mat is None:
        for top in ("model", "module", "state_dict"):
            if top in sd:
                sub = sd[top]
                for k in keys:
                    if k in sub:
                        mat = sub[k]; break
            if mat is not None:
                break
    if mat is None:
        raise KeyError("W_rec not found in checkpoint")

    if not isinstance(mat, torch.Tensor):
        raise TypeError("W_rec is not a torch.Tensor")
    mat = mat.coalesce().to(dtype=torch.float32, device="cpu")

    idx = mat.indices().cpu().numpy()                  # shape (2, nnz)
    data = mat.values().to(torch.float64).cpu().numpy()
    shape = tuple(mat.size())
    W = sp.coo_matrix((data, (idx[0], idx[1])), shape=shape)

    # ---- leak rate a ----
    a = None
    if "a" in sd: a = sd["a"]
    else:
        for top in ("model", "module", "state_dict"):
            if top in sd and "a" in sd[top]:
                a = sd[top]["a"]; break
    if a is not None:
        a = a.to(dtype=torch.float64, device="cpu").numpy()

    return W, a

def weight_stats(data):
    mean = float(np.mean(data))
    std  = float(np.std(data))
    sign_ratio = float((data > 0).mean())          # fraction positive
    skewness   = float(skew(data))
    kurt       = float(kurtosis(data))
    return mean, std, sign_ratio, skewness, kurt

def node_degree_stats(idx, n):
    """in/out degree arrays (int64)"""
    src, dst = idx
    k_out = np.bincount(src, minlength=n).astype(np.int64)
    k_in  = np.bincount(dst, minlength=n).astype(np.int64)
    return k_in, k_out

def gini(x):
    """Gini coefficient for non-negative x; returns NaN if invalid."""
    x = np.asarray(x, dtype=float)
    if x.size == 0: return np.nan
    if np.any(x < 0): x = np.abs(x)
    if np.allclose(x.sum(), 0.0): return 0.0
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    return (n + 1 - 2 * (cumx / cumx[-1]).sum()) / n

def node_sums_and_norms(idx, data, n, eps=1e-9):
    """
    per-node |in| & |out| sums and degree-normalized versions.
    Also returns sign-balance variance.
    """
    src, dst = idx
    abs_vals = np.abs(data)
    in_sum  = np.bincount(dst, weights=abs_vals, minlength=n)
    out_sum = np.bincount(src, weights=abs_vals, minlength=n)

    k_in, k_out = node_degree_stats(idx, n)

    mean_abs_in  = in_sum  / (k_in  + eps)
    mean_abs_out = out_sum / (k_out + eps)

    # sign-balance on inputs
    pos_vals = (data > 0).astype(float)
    pos_in  = np.bincount(dst, weights=pos_vals, minlength=n)
    neg_in  = np.bincount(dst, weights=1.0 - pos_vals, minlength=n)
    sb = (pos_in - neg_in) / (pos_in + neg_in + eps)

    stats = {
        "in_sum_var": float(np.var(in_sum)),
        "out_sum_var": float(np.var(out_sum)),
        "in_sum_cv": float(np.std(in_sum) / (np.mean(in_sum) + eps)),
        "out_sum_cv": float(np.std(out_sum) / (np.mean(out_sum) + eps)),
        "in_sum_gini": float(gini(in_sum)),
        "out_sum_gini": float(gini(out_sum)),

        "mean_abs_in_var": float(np.var(mean_abs_in)),
        "mean_abs_out_var": float(np.var(mean_abs_out)),
        "mean_abs_in_cv": float(np.std(mean_abs_in) / (np.mean(mean_abs_in) + eps)),
        "mean_abs_out_cv": float(np.std(mean_abs_out) / (np.mean(mean_abs_out) + eps)),

        "k_in_var": float(np.var(k_in)),
        "k_out_var": float(np.var(k_out)),
        "k_in_cv": float(np.std(k_in) / (np.mean(k_in) + eps)),
        "k_out_cv": float(np.std(k_out) / (np.mean(k_out) + eps)),

        "sign_balance_var": float(np.var(sb)),
    }
    return stats, in_sum, out_sum, mean_abs_in, mean_abs_out, k_in, k_out

def moment_trace(A: sp.csr_matrix, k=2, n_probe=32):
    """Hutch++-like: approximate trace(A^k)/n (simple randomized estimator)"""
    n = A.shape[0]
    def Av(v): return A @ v
    v = np.random.randn(n, n_probe)
    res = 0.0
    for i in range(n_probe):
        x = v[:, i:i+1]
        for _ in range(k):
            x = Av(x)
        res += float((v[:, i:i+1].T @ x))
    return res / n_probe / n

def winsorize_series(s: pd.Series, q: float) -> pd.Series:
    if q <= 0: return s
    lo, hi = s.quantile(q), s.quantile(1 - q)
    return s.clip(lower=lo, upper=hi)

def lr_residual(y, X):
    """y and X (n,d) -> residual y - X beta via least squares (adds bias)."""
    X_ = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X_, y, rcond=None)
    return y - X_ @ beta

def partial_corr(x: pd.Series, y: pd.Series, Z: pd.DataFrame) -> float:
    """Corr(x, y | Z) via residualization. Returns Pearson's r."""
    x_, y_ = x.values, y.values
    Z_ = Z.values
    rx = lr_residual(x_, Z_)
    ry = lr_residual(y_, Z_)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return np.nan
    return float(np.corrcoef(rx, ry)[0,1])

def safe_log10(s: pd.Series, eps=1e-9):
    return np.log10(np.maximum(s.astype(float), eps))

# ────────────────  MAIN  ───────────────────────────────────────────
df = pd.read_csv(args.csv)

# Optional filtering
if args.val_ppl_max is not None and "final_val_perplexity" in df.columns:
    df = df[df["final_val_perplexity"] <= args.val_ppl_max]
if args.train_ppl_max is not None and "final_train_perplexity" in df.columns:
    df = df[df["final_train_perplexity"] <= args.train_ppl_max]
df = df.copy().reset_index(drop=True)

# Metric columns
metrics_cols = [
    "rho_act",
    "fro_norm",
    "w_mean", "w_std", "w_sign_ratio", "w_skew", "w_kurt",

    # degree-unaware
    "in_sum_var", "out_sum_var", "in_sum_cv", "out_sum_cv",
    "in_sum_gini", "out_sum_gini",

    # degree-normalized (1本あたり)
    "mean_abs_in_var", "mean_abs_out_var",
    "mean_abs_in_cv", "mean_abs_out_cv",

    # degree distribution itself
    "k_in_var", "k_out_var", "k_in_cv", "k_out_cv",

    "sign_balance_var",

    # optional moments (comment-out if heavy)
    # "moment2_trace", "moment3_trace",
]
for c in metrics_cols:
    if c not in df.columns:
        df[c] = np.nan

# Compute metrics
for i, row in tqdm(df.iterrows(), total=len(df), desc="computing metrics"):
    try:
        ckpt = load_latest_pt(Path(row["model_path"]))
        W, a = fetch_mat_a(str(ckpt))
    except Exception as e:
        print(f"[skip] {row.get('model_path','?')}: {e}")
        continue

    n = W.shape[0]
    idx = np.vstack((W.row, W.col))
    data = W.data
    rho, rho2 = power_eig_radius(W.tocsr(), args.power_iters,tol=1e-6, second=False)
    # σ_max, σ_min = largest_smallest_sv(W.tocsr())
    # cond = σ_max / (σ_min + 1e-12)

    σ_max, σ_min = None, None
    cond = None
    # weights
    fro_norm = float(np.linalg.norm(data))
    w_mean, w_std, w_sign, w_sk, w_kurt = weight_stats(data)

    # node-level stats (degree-aware and normalized)
    node_dict, *_ = node_sums_and_norms(idx, data, n)

    # moments (optional; can be expensive)
    # A = W.tocsr()
    # m2 = moment_trace(A, k=2)
    # m3 = moment_trace(A, k=3)

    # assign
    df.loc[i, [
        "fro_norm", "w_mean", "w_std", "w_sign_ratio", "w_skew", "w_kurt",
        "in_sum_var", "out_sum_var", "in_sum_cv", "out_sum_cv",
        "in_sum_gini", "out_sum_gini",
        "mean_abs_in_var", "mean_abs_out_var",
        "mean_abs_in_cv", "mean_abs_out_cv",
        "k_in_var", "k_out_var", "k_in_cv", "k_out_cv",
        "sign_balance_var",
        # "moment2_trace", "moment3_trace"
    ]] = [
        rho,
        fro_norm, w_mean, w_std, w_sign, w_sk, w_kurt,
        node_dict["in_sum_var"], node_dict["out_sum_var"],
        node_dict["in_sum_cv"], node_dict["out_sum_cv"],
        node_dict["in_sum_gini"], node_dict["out_sum_gini"],
        node_dict["mean_abs_in_var"], node_dict["mean_abs_out_var"],
        node_dict["mean_abs_in_cv"], node_dict["mean_abs_out_cv"],
        node_dict["k_in_var"], node_dict["k_out_var"],
        node_dict["k_in_cv"], node_dict["k_out_cv"],
        node_dict["sign_balance_var"],
        # m2, m3
    ]

# Save metrics
df.to_csv(args.out_csv, index=False)
print(f"✅ metrics appended → {args.out_csv}")

# Prepare PPL columns
ppl_cols = [c for c in ["final_train_perplexity", "final_val_perplexity", "msf_ppl"] if c in df.columns]
if args.log_ppl:
    for c in ppl_cols:
        df[f"log10_{c}"] = safe_log10(df[c])
    ppl_cols_plot = [f"log10_{c}" for c in ppl_cols]
else:
    ppl_cols_plot = ppl_cols

# Optional winsorization to reduce outlier leverage (applies to all used columns)
if args.winsor_q > 0:
    for c in metrics_cols + ppl_cols_plot:
        if c in df.columns:
            df[c] = winsorize_series(df[c], args.winsor_q)

# ────────────────  Correlations  ──────────────────────────────────
def compute_corr_tables(df: pd.DataFrame, metrics, ppl_targets, controls=None):
    """Return three DataFrames: Pearson, Spearman, Partial (if controls)."""
    pearson = pd.DataFrame(index=metrics, columns=ppl_targets, dtype=float)
    spearm  = pd.DataFrame(index=metrics, columns=ppl_targets, dtype=float)
    partial = pd.DataFrame(index=metrics, columns=ppl_targets, dtype=float) if controls is not None else None

    for m in metrics:
        for p in ppl_targets:
            s = df[[m, p]].dropna()
            if len(s) < 5:
                pearson.loc[m,p] = np.nan
                spearm.loc[m,p]  = np.nan
                if partial is not None: partial.loc[m,p] = np.nan
                continue
            pearson.loc[m,p] = float(np.corrcoef(s[m], s[p])[0,1])
            spearm.loc[m,p]  = float(spearmanr(s[m], s[p]).correlation)

            if controls is not None:
                cols = [c for c in controls if c in df.columns]
                if cols:
                    S = df[[m, p] + cols].dropna()
                    if len(S) >= 5:
                        Z = S[cols]
                        partial.loc[m,p] = partial_corr(S[m], S[p], Z)
                    else:
                        partial.loc[m,p] = np.nan
    return pearson, spearm, partial

control_cols = ["k_in_var", "k_out_var"]  # control for degree dispersion
pearson, spearm, partial = compute_corr_tables(df, metrics_cols, ppl_cols_plot, controls=control_cols)

def save_heatmap(mat: pd.DataFrame, title: str, out_name: str):
    if mat is None: return
    fig, ax = plt.subplots(figsize=(max(10, 0.6*len(mat.columns)), max(8, 0.5*len(mat.index))))
    im = ax.imshow(mat.values.astype(float), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(mat.columns))); ax.set_xticklabels(mat.columns, rotation=90)
    ax.set_yticks(range(len(mat.index)));   ax.set_yticklabels(mat.index)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(plots_dir / out_name, dpi=160)
    plt.close()
def draw_boxplot(series_list, x_labels, ylabel, title, out_path, yscale="linear"):
    if not series_list: return
    plt.figure(figsize=(max(10, len(series_list)*0.5), 6))
    plt.boxplot(series_list, positions=range(len(series_list)), showfliers=False)
    plt.xticks(range(len(series_list)), x_labels, rotation=90)
    plt.xlabel("runs (sorted by target)"); plt.ylabel(ylabel)
    plt.title(title)
    ax = plt.gca()
    ax.set_yscale(yscale)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
# save tables + heatmaps
pearson.to_csv(plots_dir / "corr_tables/pearson.csv")
spearm.to_csv(plots_dir / "corr_tables/spearman.csv")
if partial is not None:
    partial.to_csv(plots_dir / "corr_tables/partial_given_k.csv")

save_heatmap(pearson, "Pearson correlation (metrics ↔ targets)", "corr_heatmap_pearson.png")
save_heatmap(spearm,  "Spearman correlation (metrics ↔ targets)", "corr_heatmap_spearman.png")
if partial is not None:
    save_heatmap(partial, "Partial corr | degree vars (metrics ↔ targets)", "corr_heatmap_partial_given_k.png")

print("📈 correlation heat-maps saved.")
@lru_cache(maxsize=32)
def fetch_mat_a(ckpt_path: str):
    """Return scipy.sparse.coo_matrix (float64) & leak-rate numpy array (float64)"""
    sd = torch.load(ckpt_path, map_location="cpu")

    # ---- W_rec を探す（直下 or model/module/state_dict 配下） ----
    mat = None
    for k in ("W_rec", "w_rec", "W_rec_coo"):
        if k in sd:
            mat = sd[k]; break
    if mat is None:
        for top in ("model", "module", "state_dict"):
            if top in sd:
                sub = sd[top]
                for k in ("W_rec", "w_rec", "W_rec_coo"):
                    if k in sub:
                        mat = sub[k]; break
            if mat is not None:
                break
    if mat is None:
        raise KeyError("W_rec not found in checkpoint")

    # ---- COO にまとめて dtype を float32/float64 に統一（bfloat16 対策） ----
    if not isinstance(mat, torch.Tensor):
        raise TypeError("W_rec is not a torch.Tensor")
    mat = mat.coalesce().to(dtype=torch.float32, device="cpu")   # ← ここが重要

    idx = mat.indices().cpu().numpy()                            # shape (2, nnz), int64
    data = mat.values().to(torch.float64).cpu().numpy()          # ← float64 に上げる
    shape = tuple(mat.size())

    # ---- scipy COO に変換 ----
    W = sp.coo_matrix((data, (idx[0], idx[1])), shape=shape)

    # ---- leak rate a ----
    a = None
    for k in ("a",):
        if k in sd: a = sd[k]
    if a is None:
        for top in ("model", "module", "state_dict"):
            if top in sd and "a" in sd[top]:
                a = sd[top]["a"]
                break
    if a is not None:
        a = a.to(dtype=torch.float64, device="cpu").numpy()

    return W, a

import numpy as np
import scipy.sparse as sp

def power_eig_radius(
    A: sp.csr_matrix,
    max_iter: int = 200,
    tol: float = 1e-6,
    second: bool = False,
):
    """
    Parameters
    ----------
    A : sp.csr_matrix
        対象行列（疎・正方）
    max_iter : int
        最大イテレーション数 (fallback の上限)
    tol : float
        ρ の相対更新量が `tol` 未満になったら収束とみなす
    second : bool
        True のとき第2固有値 |λ₂| も deflation で推定する

    Returns
    -------
    rho1 : float
        最大固有値絶対値 (= スペクトル半径)
    rho2 : float | None
        |λ₂|（second=True のときのみ）
    iters1 : int
        ρ₁ の収束に要したイテレーション数
    iters2 : int | None
        ρ₂ の収束イテレーション（second=True のとき）
    """
    n = A.shape[0]

    # ── λ₁ ─────────────────────────────────────────────
    v = np.random.randn(n, 1)
    v /= np.linalg.norm(v)
    rho_old = 0.0
    with tqdm(total=max_iter, desc="power iteration") as pbar:
        for k in range(1, max_iter + 1):
            pbar.update(1)
            v = A @ v
            v /= np.linalg.norm(v)
            rho = float(abs((A @ v).T @ v)[0, 0])
            if abs(rho - rho_old) / (rho_old + 1e-12) < tol:
                break
            rho_old = rho
    rho1, iters1 = rho, k

    if not second:
        return rho1, None

    # ── λ₂ (deflation) ────────────────────────────────
    Av   = A @ v
    Adef = A - (Av @ v.T)            # rank-1 deflation

    w = np.random.randn(n, 1)
    w -= v @ (v.T @ w)               # v と直交に
    w /= np.linalg.norm(w)
    rho_old = 0.0
    with tqdm(total=max_iter, desc="power iteration") as pbar:
        for k in range(1, max_iter + 1):
            pbar.update(1)
            w = Adef @ w
            # v 方向成分を都度引いて直交性維持
            w -= v @ (v.T @ w)
            w /= np.linalg.norm(w)
            rho = float(abs((A @ w).T @ w)[0, 0])
            if abs(rho - rho_old) / (rho_old + 1e-12) < tol:
                break
            rho_old = rho
    rho2, iters2 = rho, k

    return rho1, rho2, iters1, iters2
# ────────────────  Scatter plots  ─────────────────────────────────
def scatter_with_fit(x, y, xlabel, ylabel, title, out_path):
    s = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(s) < 5: return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(s["x"], s["y"], alpha=0.6)
    # OLS line
    X = np.column_stack([np.ones(len(s)), s["x"].values])
    b, *_ = np.linalg.lstsq(X, s["y"].values, rcond=None)
    xline = np.linspace(s["x"].min(), s["x"].max(), 100)
    yline = b[0] + b[1]*xline
    ax.plot(xline, yline, linewidth=2)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

for metric in metrics_cols:
    for ppl_col in ppl_cols_plot:
        if metric not in df.columns or ppl_col not in df.columns: continue
        scatter_with_fit(
            df[metric], df[ppl_col],
            xlabel=metric, ylabel=ppl_col,
            title=f"{metric} vs {ppl_col}",
            out_path=plots_dir / f"scatter_{metric}_vs_{ppl_col}.png"
        )
print("📊 scatter plots saved.")

# ────────────────  Box Plot (weights vs perplexity) ───────────────
if args.make_boxplot and "final_val_perplexity" in df.columns:
    ppl_col = "final_val_perplexity" if not args.log_ppl else "log10_final_val_perplexity"
    df_sorted = df.dropna(subset=[ppl_col]).sort_values(by=ppl_col)

    all_weights, x_labels = [], []
    for _, row in tqdm(df_sorted.iterrows(), total=len(df_sorted), desc="collect weights for boxplot"):
        try:
            ckpt = load_latest_pt(Path(row["model_path"]))
            W, a = fetch_mat_a(str(ckpt))
            all_weights.append(W.data)
            x_labels.append(f"{row[ppl_col]:.2f}")
        except Exception as e:
            print(f"[skip] {row.get('model_path','?')}: {e}")
            continue

    if all_weights:
        plt.figure(figsize=(max(10, len(all_weights)*0.5), 6))
        plt.boxplot(all_weights, positions=range(len(all_weights)), showfliers=False)
        plt.xticks(range(len(all_weights)), x_labels, rotation=90)
        plt.xlabel(ppl_col); plt.ylabel("Weight values")
        plt.title(f"Weight distributions (box plot) sorted by {ppl_col}")
        plt.tight_layout()
        boxplot_path = plots_dir / f"weights_boxplot_sorted_by_{ppl_col}.png"
        plt.savefig(boxplot_path, dpi=150); plt.close()
        print(f"📦 box plot saved → {boxplot_path}")
# ────────────────  Box Plot (node sums vs perplexity) ─────────────
if "final_val_perplexity" in df.columns:
    ppl_col = "final_val_perplexity" if not args.log_ppl else "log10_final_val_perplexity"
    df_sorted = df.dropna(subset=[ppl_col]).sort_values(by=ppl_col)

    in_sums_list, out_sums_list = [], []
    mean_abs_in_list, mean_abs_out_list = [], []
    x_labels = []

    for _, row in tqdm(df_sorted.iterrows(), total=len(df_sorted), desc="collect node sums for boxplot"):
        try:
            ckpt = load_latest_pt(Path(row["model_path"]))
            W, a = fetch_mat_a(str(ckpt))
            n = W.shape[0]
            idx = np.vstack((W.row, W.col))
            data = W.data

            # 既存ユーティリティを活用
            stats, in_sum, out_sum, mean_abs_in, mean_abs_out, _, _ = node_sums_and_norms(idx, data, n)

            in_sums_list.append(in_sum)
            out_sums_list.append(out_sum)
            mean_abs_in_list.append(mean_abs_in)
            mean_abs_out_list.append(mean_abs_out)

            x_labels.append(f"{row[ppl_col]:.2f}")
        except Exception as e:
            print(f"[skip sums] {row.get('model_path','?')}: {e}")
            continue

    # 生の絶対値和（ノードごと合計）
    draw_boxplot(
        in_sums_list, x_labels,
        ylabel="sum_j |w_{ji}| (per node)",
        title=f"Input strength sums per node (sorted by {ppl_col})",
        out_path=plots_dir / f"node_in_sums_boxplot_sorted_by_{ppl_col}.png",
        yscale=args.box_yscale
    )
    draw_boxplot(
        out_sums_list, x_labels,
        ylabel="sum_j |w_{ij}| (per node)",
        title=f"Output strength sums per node (sorted by {ppl_col})",
        out_path=plots_dir / f"node_out_sums_boxplot_sorted_by_{ppl_col}.png",
        yscale=args.box_yscale
    )

    # 次数で割った平均絶対重み（1本あたりの強さ）
    draw_boxplot(
        mean_abs_in_list, x_labels,
        ylabel="mean |w_{ji}| per incoming edge",
        title=f"Degree-normalized input strength (sorted by {ppl_col})",
        out_path=plots_dir / f"node_mean_abs_in_boxplot_sorted_by_{ppl_col}.png",
        yscale=args.box_yscale
    )
    draw_boxplot(
        mean_abs_out_list, x_labels,
        ylabel="mean |w_{ij}| per outgoing edge",
        title=f"Degree-normalized output strength (sorted by {ppl_col})",
        out_path=plots_dir / f"node_mean_abs_out_boxplot_sorted_by_{ppl_col}.png",
        yscale=args.box_yscale
    )

print("✅ done.")