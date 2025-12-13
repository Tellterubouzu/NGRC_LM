
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
ap.add_argument("--csv", default="run_table.csv",
                help="Input CSV (requires model_path column)")
ap.add_argument("--out_csv", default="run_table_metrics.csv")
ap.add_argument("--plots_dir", default="plots")
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

from functools import lru_cache

def _iter_tensor_items(d, prefix=""):
    for k, v in d.items():
        name = f"{prefix}.{k}" if prefix else str(k)
        if torch.is_tensor(v):
            yield name, v
        elif isinstance(v, dict):
            yield from _iter_tensor_items(v, name)

def _pick_reservoir(items):
    # (N,N) を候補にし名前ヒントでスコア
    hints = ("rec", "reservoir", "recurrent", "adj", "matrix", "w_rec", "W_rec", "W_rec_T")
    best = None; score_best = -1
    for name, t in items:
        try:
            if t.ndim != 2: continue
            n, m = t.shape
            if n != m or n < 16:  # 小さすぎ/非正方は除外
                continue
            s = sum(name.lower().count(h) for h in hints) + int(min(n,4096)/256)
            if s > score_best:
                best, score_best = (name, t), s
        except:
            pass
    return best  # (name, tensor) or None

@lru_cache(maxsize=64)
def fetch_mat_a(ckpt_path: str):
    sd = torch.load(ckpt_path, map_location="cpu")
    # よくあるラッパを優先的に探索
    dicts = []
    if isinstance(sd, dict):
        for k in ("state_dict","model_state_dict","model","module","net"):
            if k in sd and isinstance(sd[k], dict):
                dicts.append(sd[k])
        dicts.append(sd)
    else:
        raise TypeError(f"checkpoint is not dict-like: {type(sd)}")

    # まずはキー名で直撃（高速経路）
    for d in dicts:
        for k in ("W_rec", "w_rec", "W_rec_coo", "W_rec_T"):
            if k in d and torch.is_tensor(d[k]) and d[k].ndim == 2:
                t = d[k].to("cpu")
                if k.endswith("_T"):  # 転置で保存されている場合は元に戻す
                    t = t.t().contiguous()
                arr = t.to(dtype=torch.float32).cpu()
                if getattr(arr, "is_sparse", False) or getattr(arr, "is_sparse_csr", False):
                    arr = arr.to_sparse_coo().coalesce()
                    idx = arr.indices().cpu().numpy()
                    data = arr.values().to(torch.float64).cpu().numpy()
                    n = int(arr.size(0))
                    W = sp.coo_matrix((data, (idx[0], idx[1])), shape=(n, n))
                else:
                    W = sp.coo_matrix(arr.numpy())
                break
        else:
            continue
        break
    else:
        # 汎用探索（(N,N)のテンソルから最適候補）
        cand = None
        for d in dicts:
            cand = _pick_reservoir(list(_iter_tensor_items(d)))
            if cand: break
        if not cand:
            raise KeyError("Reservoir matrix not found (no square NxN tensor).")
        name, t = cand
        t = t.to("cpu")
        if getattr(t, "is_sparse", False) or getattr(t, "is_sparse_csr", False):
            t = t.to_sparse_coo().coalesce().to(dtype=torch.float32)
            idx = t.indices().cpu().numpy()
            data = t.values().to(torch.float64).cpu().numpy()
            n = int(t.size(0))
            W = sp.coo_matrix((data, (idx[0], idx[1])), shape=(n, n))
        else:
            W = sp.coo_matrix(t.to(dtype=torch.float32).cpu().numpy())

    # leak rate a（あれば）
    a = None
    for d in dicts:
        for key in ("a", "leak", "leak_rate", "alpha"):
            v = d.get(key) if isinstance(d, dict) else None
            if torch.is_tensor(v):
                a = v.to(torch.float64).cpu().numpy()
                break
        if a is not None: break

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

    
    cols = [
        "rho_act",
        "fro_norm", "w_mean", "w_std", "w_sign_ratio", "w_skew", "w_kurt",
        "in_sum_var", "out_sum_var", "in_sum_cv", "out_sum_cv",
        "in_sum_gini", "out_sum_gini",
        "mean_abs_in_var", "mean_abs_out_var",
        "mean_abs_in_cv", "mean_abs_out_cv",
        "k_in_var", "k_out_var", "k_in_cv", "k_out_cv",
        "sign_balance_var",
    ]
    vals = [
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
    ]
    assert len(cols) == len(vals)
    df.loc[i, cols] = vals
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
    #X = np.column_stack([np.ones(len(s)), s["x"].values])
    #b, *_ = np.linalg.lstsq(X, s["y"].values, rcond=None)
    # xline = np.linspace(s["x"].min(), s["x"].max(), 100)
    # yline = b[0] + b[1]*xline
    # ax.plot(xline, yline, linewidth=2)
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
