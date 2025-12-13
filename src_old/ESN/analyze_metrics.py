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

# ────────────────  CLI  ────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="run_table_65536_0827.csv",
                help="Input CSV (requires model_path column)")
ap.add_argument("--out_csv", default="run_table_metrics_except_execution_4096.csv")
ap.add_argument("--plots_dir", default="plots_65536a_0831")
ap.add_argument("--power_iters", type=int, default=20000,
                help="Iterations for power method (eigen / singular)")
ap.add_argument("--sample_nodes", type=int, default=2048,
                help="For huge graphs, BFS sample size for avg shortest path")
args = ap.parse_args()

plots_dir = Path(args.plots_dir)
plots_dir.mkdir(exist_ok=True)

# ────────────────  UTILS  ──────────────────────────────────────────

def load_latest_pt(path: Path) -> Path:
    """path が .pt ファイルならそのまま、フォルダなら再帰で最新 .pt を返す"""
    print(path)
    if path.is_file() and path.suffix == ".pt":
        return path
    if path.is_dir():
        pts = sorted(path.rglob("*.pt"),        # 再帰探索
                     key=lambda p: p.stat().st_mtime,
                     reverse=True)
        if pts:
            return pts[0]
    raise FileNotFoundError(f"No .pt file found under {path}")
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
# def fetch_mat_a(ckpt_path: str):
#     """Return sparse COO matrix (scipy) & leak-rate numpy array"""
#     sd = torch.load(ckpt_path, map_location="cpu")
#     for k in ("W_rec", "w_rec"):
#         if k in sd:           mat = sd[k]
#         elif "model" in sd and k in sd["model"]:  mat = sd["model"][k]
#         elif "module" in sd and k in sd["module"]:mat = sd["module"][k]
#         else:                 continue
#         break
#     else:
#         raise KeyError("W_rec not found")

#     if isinstance(mat, torch.Tensor): mat = mat.coalesce()
#     idx   = mat.indices().numpy()
#     data  = mat.values().numpy().astype(np.float64)
#     shape = mat.size()
#     W     = sp.coo_matrix((data, idx), shape=shape)

#     for k in ("a",):
#         if k in sd:               a = sd[k]
#         elif "model" in sd and k in sd["model"]:   a = sd["model"][k]
#         elif "module" in sd and k in sd["module"]: a = sd["module"][k]
#         else:                     a = None
#     if a is not None: a = a.cpu().numpy().astype(np.float64)

#     return W, a
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
# def power_eig_radius(A: sp.csr_matrix, iters=200, second=False):
#     """return ρ_max (and optionally ρ_2) via power iteration / deflation"""
#     n = A.shape[0]
#     v = np.random.randn(n, 1)
#     v /= np.linalg.norm(v)
#     for _ in range(iters):
#         v = A @ v
#         v /= np.linalg.norm(v)
#     lam1 = float(np.abs((A @ v).T @ v))
#     if not second:
#         return lam1, None
#     # deflation for λ₂ magnitude
#     Av = (A @ v)
#     A_def = A - (Av @ v.T)  # rank-1 deflation
#     w = np.random.randn(n, 1)
#     w -= v @ (v.T @ w)
#     w /= np.linalg.norm(w)
#     for _ in range(iters):
#         w = A_def @ w
#         w -= v @ (v.T @ w)  # keep orthogonal
#         w /= np.linalg.norm(w)
#     lam2 = float(np.abs((A @ w).T @ w))
#     return lam1, lam2

def largest_smallest_sv(A: sp.csr_matrix, iters=200):
    """largest σ_max and smallest σ_min using svds (k=1) twice"""
    σ_max = spla.svds(A, k=1, return_singular_vectors=False)[0]
    σ_min = spla.svds(A, k=1, return_singular_vectors=False, which="SM")[0]
    return float(σ_max), float(σ_min)

def weight_stats(data):
    mean = float(data.mean())
    std  = float(data.std())
    sign_ratio = float((data > 0).mean())          # fraction positive
    skewness   = float(skew(data))
    kurt       = float(kurtosis(data))
    return mean, std, sign_ratio, skewness, kurt

def self_loops(idx):
    return int(np.sum(idx[0] == idx[1]))

# def graph_metrics(idx, n_nodes, sample_nodes=2048):
#     """Return SCC stats, avg shortest path, diameter, degree stats"""
#     G = nx.DiGraph()
#     G.add_nodes_from(range(n_nodes))
#     G.add_edges_from(zip(idx[0], idx[1]))

#     # SCC sizes
#     sccs = [len(c) for c in nx.strongly_connected_components(G)]
#     max_scc = max(sccs)
#     n_scc   = len(sccs)

#     # undirected largest component for path stats
#     UG = G.to_undirected()
#     largest_cc = max(nx.connected_components(UG), key=len)
#     H = UG.subgraph(largest_cc).copy()
#     # sample nodes if too large
#     if len(H) > sample_nodes:
#         import random
#         sample = random.sample(list(H.nodes), sample_nodes)
#         H = H.subgraph(sample).copy()

#     try:
#         avg_sp = nx.average_shortest_path_length(H)
#         diam   = nx.diameter(H)
#     except nx.NetworkXError:
#         avg_sp = np.nan
#         diam   = np.nan

#     degs = np.array([d for _, d in H.degree()])
#     deg_skew = float(skew(degs))
#     deg_var  = float(degs.var())

#     return max_scc, n_scc, avg_sp, diam, deg_skew, deg_var
import numpy as np
import networkx as nx
import random
from scipy.stats import skew

def graph_metrics(idx, n_nodes, sample_nodes=2048):
    """
    idx: 2×nnz の ndarray ([rows, cols])
    n_nodes: 全ノード数 N
    sample_nodes: サブサンプル時のノード数上限
    returns:
      scc_max, scc_count,
      avg_shortest_path, diameter,
      deg_skew, deg_var
    """
    # 有向グラフを作成
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    if idx.size > 0:
        G.add_edges_from(zip(idx[0], idx[1]))

    # ── 強連結成分 (SCC) ─────────────────────────────
    sccs = list(nx.strongly_connected_components(G))
    scc_sizes = [len(c) for c in sccs]
    scc_max   = max(scc_sizes) if scc_sizes else 0
    scc_count = len(scc_sizes)

    # ── 無向化して最大連結成分を取る ────────────────────
    UG = G.to_undirected()
    comps = list(nx.connected_components(UG))
    if not comps:
        # ノードはいるがエッジが一切ない場合
        avg_sp, diam = np.nan, np.nan
        deg_skew, deg_var = np.nan, np.nan
    else:
        # 最大成分
        largest = max(comps, key=len)
        H = UG.subgraph(largest).copy()
        # ノード数が多すぎるときはサンプリング
        if H.number_of_nodes() > sample_nodes:
            sample = random.sample(list(H.nodes), sample_nodes)
            H = UG.subgraph(sample).copy()

        # 平均最短経路 & 直径
        if H.number_of_nodes() <= 1:
            avg_sp, diam = 0.0, 0.0
        else:
            # 念のため再チェック
            if nx.is_connected(H):
                avg_sp = nx.average_shortest_path_length(H)
                diam   = nx.diameter(H)
            else:
                avg_sp, diam = np.nan, np.nan

        # 次数分布統計
        degs = np.fromiter((d for _, d in H.degree()), dtype=float)
        deg_skew = skew(degs) if degs.size>0 else np.nan
        deg_var  = degs.var() if degs.size>0 else np.nan

    return scc_max, scc_count, avg_sp, diam, deg_skew, deg_var
def moment_trace(A: sp.csr_matrix, k=2, n_probe=32):
    """Hutch++: approximate trace(A^k) / n"""
    n = A.shape[0]
    def Av(v): return A @ v
    v = np.random.randn(n, n_probe)
    res = 0.0
    for i in range(n_probe):
        x = v[:, i:i+1]
        for _ in range(k):
            x = Av(x)
        res += float((v[:, i:i+1].T @ x))
    return res / n_probe / n    # normalize by n

def node_sums(idx, data, n):
    """per-node |in| & |out| sums, sign-balance"""
    src, dst = idx
    abs_vals = np.abs(data)
    in_sum  = np.bincount(dst, weights=abs_vals, minlength=n)
    out_sum = np.bincount(src, weights=abs_vals, minlength=n)
    pos_vals = data > 0
    pos_in  = np.bincount(dst, weights=pos_vals, minlength=n)
    neg_in  = np.bincount(dst, weights=~pos_vals, minlength=n)
    # sign balance   (pos - neg) / (pos + neg + ε)
    ε = 1e-9
    sb = (pos_in - neg_in) / (pos_in + neg_in + ε)
    return (float(in_sum.var()),
            float(out_sum.var()),
            float(sb.var()))

# ────────────────  MAIN LOOP  ──────────────────────────────────────
df = pd.read_csv(args.csv)
metrics_cols = [
    #"rho_act", "eig_gap", "sigma_max", "sigma_min", "cond",
    "rho_act",
    "fro_norm",
    "w_mean", "w_std", "w_sign_ratio", "w_skew", "w_kurt",
    "self_loops",
    #"scc_max", "scc_count",
    #"avg_shortest_path", " diameter",
    "deg_skew", "deg_var",
    "moment2_trace", "moment3_trace",
    "in_sum_var", "out_sum_var", "sign_balance_var",
]
for c in metrics_cols:
    df[c] = np.nan

for i, row in tqdm(df.iterrows(), total=len(df), desc="computing metrics"):
    try:
        ckpt = load_latest_pt(Path(row["model_path"]))
        print(ckpt)
        W, a = fetch_mat_a(str(ckpt))
    except Exception as e:
        print(e)
        continue
    if row["final_val_perplexity"] > 1000:
        continue
    #if row["val_ppl"]
    n = W.shape[0]
    idx = np.vstack((W.row, W.col))
    data = W.data
    fro_norm = np.linalg.norm(data)
    # ---------- eigen / singular ----------
    print("eigen / singular")
    rho, rho2 = power_eig_radius(W.tocsr(), args.power_iters,tol=1e-6, second=False)
    # σ_max, σ_min = largest_smallest_sv(W.tocsr())
    # cond = σ_max / (σ_min + 1e-12)

    σ_max, σ_min = None, None
    cond = None
    

    # ---------- weight stats ----------
    print("weight stats")
    w_mean, w_std, w_sign, w_sk, w_kurt = weight_stats(data)

    # ---------- graph ----------
    print("graph")
    sl = self_loops(idx)
    scc_max, scc_cnt, avg_sp, diam, deg_sk, deg_var = graph_metrics(
        idx, n, args.sample_nodes
    )
    print(scc_max, scc_cnt, avg_sp, diam, deg_sk, deg_var)

    # ---------- moments ----------
    print("moments")
    m2 = moment_trace(W.tocsr(), k=2)
    m3 = moment_trace(W.tocsr(), k=3)

    # ---------- node sums ----------
    print("node sums")
    in_var, out_var, sb_var = node_sums(idx, data, n)

    # ---------- assign ----------
    df.loc[i, metrics_cols] = [
        #rho, (rho - rho2) if rho2 else np.nan, σ_max, σ_min, cond,
        rho,
        np.linalg.norm(data),
        w_mean, w_std, w_sign, w_sk, w_kurt,
        sl, #scc_max, scc_cnt,
        #avg_sp, diam,
        deg_sk, deg_var,
        m2, m3,
        in_var, out_var, sb_var,
    ]        

# ────────────────  SAVE  ──────────────────────────────────────────
df.to_csv(args.out_csv, index=False)
print(f"✅ metrics appended → {args.out_csv}")

# ────────────────  CORRELATION HEAT-MAP  ─────────────────────────
corr_cols = metrics_cols + ["final_train_perplexity", "final_val_perplexity", "msf_ppl"]
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

# ────────────────  各指標 vs ppl の散布図を作成・保存 ─────────────────────────
for metric in metrics_cols:
    for ppl_col in ["final_train_perplexity", "final_val_perplexity", "msf_ppl"]:
        plt.figure()
        plt.scatter(df[metric], df[ppl_col], alpha=0.6)
        plt.xlabel(metric)
        plt.ylabel(ppl_col)
        plt.title(f"{metric} vs {ppl_col}")
        plt.tight_layout()
        scatter_path = plots_dir / f"{metric}_vs_{ppl_col}.png"
        plt.savefig(scatter_path, dpi=150)
        plt.close()
        print(f"📊 散布図を保存 → {scatter_path}")
