#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ESN Reservoir Analyzer with Boxplot
# -----------------------------------
# - CSV を読み込み、.pt から W_rec を抽出して安定性指標を集計
# - 追加: final_val_perplexity の箱ひげ図を Matplotlib で出力（--boxplot_out）
#
# 使い方:
#   python esn_reservoir_analysis.py \
#     --runs_csv runs.csv \
#     --out_csv reservoir_metrics.csv \
#     --device cpu \
#     --power_iters 50 \
#     --hutch_samples 8 \
#     --boxplot_out val_ppl_boxplot.png \
#     --boxplot_from runs.csv \
#     --boxplot_yscale log

import argparse
import csv
import math
import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, Any, List

import torch

def log(msg: str):
    print(msg, file=sys.stderr)

# ---------- Sparse helpers ----------

def to_sparse_tensor(obj: torch.Tensor) -> torch.Tensor:
    """Ensure a tensor is in a sparse layout (COO or CSR)."""
    if obj.layout in (torch.sparse_coo, torch.sparse_csr):
        return obj.coalesce() if obj.layout == torch.sparse_coo else obj
    if getattr(obj, "is_sparse", False):
        # Older sparse type
        return obj.coalesce()
    # Dense -> convert to COO (warning for huge matrices)
    log("[warn] dense tensor detected; converting to COO (may be huge).")
    coo = obj.to_sparse_coo()
    return coo.coalesce()

def spmv(W: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Sparse/dense matrix-vector multiply."""
    if W.layout == torch.sparse_csr:
        return torch.sparse.mm(W, v.unsqueeze(1)).squeeze(1)
    elif W.layout == torch.sparse_coo:
        return torch.sparse.mm(W, v.unsqueeze(1)).squeeze(1)
    else:
        return W @ v

def spmm(W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Sparse/dense matrix-matrix multiply."""
    if W.layout == torch.sparse_csr:
        return torch.sparse.mm(W, X)
    elif W.layout == torch.sparse_coo:
        return torch.sparse.mm(W, X)
    else:
        return W @ X

def sptm(W: torch.Tensor) -> torch.Tensor:
    """Sparse/dense transpose preserving layout as much as possible."""
    if W.layout == torch.sparse_csr:
        return W.transpose(0, 1)
    elif W.layout == torch.sparse_coo:
        return W.transpose(0, 1).coalesce()
    else:
        return W.t()

def nnz_of_sparse(W: torch.Tensor) -> int:
    if W.layout == torch.sparse_csr:
        return int(W.nnz())
    elif W.layout == torch.sparse_coo:
        return int(W._nnz())
    else:
        return int((W != 0).sum().item())

def size_of_sparse(W: torch.Tensor) -> Tuple[int, int]:
    return W.shape[0], W.shape[1]

# ---------- W_rec extraction ----------

RESERVOIR_KEY_HINTS = [
    "reservoir.W_rec", "reservoir.W", "W_rec", "Wres", "W_res", "rec.weight",
]

INDICES_VALUE_PATTERNS = [
    (r"(.*)W[_\.]?rec[_\.]?(indices)$", r"\1W_rec_values", r"\1W_rec_size"),
    (r"(.*)W[_\.]?rec\.indices$", r"\1W_rec.values", r"\1W_rec.size"),
    (r"(.*)recurrence\.indices$", r"\1recurrence.values", r"\1recurrence.size"),
]

def build_sparse_from_indices_values(state: Dict[str, Any]) -> Optional[torch.Tensor]:
    for k in list(state.keys()):
        if k.endswith("_indices") or k.endswith(".indices"):
            idx_key = k
            for pat_idx, pat_val, pat_size in INDICES_VALUE_PATTERNS:
                m = re.match(pat_idx, idx_key)
                if m:
                    val_key = m.expand(pat_val)
                    size_key = m.expand(pat_size)
                    if val_key in state and size_key in state:
                        indices = state[idx_key]
                        values = state[val_key]
                        size = state[size_key]
                        if isinstance(size, (list, tuple)):
                            size = torch.Size(size)
                        elif isinstance(size, torch.Tensor):
                            size = torch.Size(size.tolist())
                        coo = torch.sparse_coo_tensor(indices, values, size=size)
                        return coo.coalesce()
    return None

def guess_reservoir_key(state: Dict[str, Any]) -> Optional[str]:
    for key in RESERVOIR_KEY_HINTS:
        if key in state:
            t = state[key]
            if isinstance(t, torch.Tensor) and t.ndim == 2 and t.shape[0] == t.shape[1]:
                return key
    candidates = []
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and v.ndim == 2 and v.shape[0] == v.shape[1]:
            if any(s in k.lower() for s in ["rec", "reservoir", "recur", "w_rec", "wres", "w_res"]):
                n = v.shape[0]
                candidates.append((n, k))
    if not candidates:
        for k, v in state.items():
            if isinstance(v, torch.Tensor) and v.ndim == 2 and v.shape[0] == v.shape[1]:
                n = v.shape[0]
                candidates.append((n, k))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    return None

@dataclass
class ReservoirMatrices:
    W_rec: torch.Tensor
    leak: Optional[torch.Tensor] = None

def extract_W_rec_and_leak(state: Dict[str, Any], device: torch.device) -> ReservoirMatrices:
    coo = build_sparse_from_indices_values(state)
    leak = None
    if coo is not None:
        W = to_sparse_tensor(coo).to(device)
    else:
        key = guess_reservoir_key(state)
        if key is None:
            raise KeyError("Could not locate W_rec. Please adjust key hints.")
        W0 = state[key]
        if not isinstance(W0, torch.Tensor):
            raise TypeError(f"Key {key} is not a tensor.")
        W = to_sparse_tensor(W0.to(device))
    for lk in ["a", "leak", "leaky_rate", "alpha", "reservoir.a", "reservoir.alpha"]:
        if lk in state and isinstance(state[lk], torch.Tensor):
            leak = state[lk].to(device).flatten()
            break
    return ReservoirMatrices(W_rec=W, leak=leak)

# ---------- Metrics: linear algebra ----------

@torch.no_grad()
def estimate_spectral_norm(W: torch.Tensor, iters: int = 50, device: Optional[torch.device] = None) -> float:
    m, n = size_of_sparse(W)
    if device is None:
        device = W.device
    v = torch.randn(n, device=device)
    v = v / (v.norm() + 1e-12)
    WT = sptm(W)
    for _ in range(iters):
        u = spmv(W, v)
        v = spmv(WT, u)
        nrm = v.norm()
        if not torch.isfinite(nrm):
            return float('nan')
        v = v / (nrm + 1e-12)
    u = spmv(W, v)
    sigma = u.norm().item()
    return float(sigma)

@torch.no_grad()
def estimate_spectral_radius_power(W: torch.Tensor, iters: int = 50, device: Optional[torch.device] = None) -> float:
    n, _ = size_of_sparse(W)
    if device is None:
        device = W.device
    v = torch.randn(n, device=device)
    v = v / (v.norm() + 1e-12)
    lam_abs = 0.0
    for _ in range(iters):
        w = spmv(W, v)
        nrm = w.norm()
        if nrm < 1e-20 or not torch.isfinite(nrm):
            break
        v = w / (nrm + 1e-12)
        lam_abs = nrm.item()
    return float(lam_abs)

@torch.no_grad()
def gershgorin_upper_bound(W: torch.Tensor) -> float:
    n, _ = size_of_sparse(W)
    if W.layout == torch.sparse_csr:
        crow = W.crow_indices()
        col = W.col_indices()
        val = W.values()
        absval = val.abs()
        max_bound = 0.0
        for i in range(n):
            start = int(crow[i].item())
            end = int(crow[i+1].item())
            row_cols = col[start:end]
            row_vals = absval[start:end]
            diag_mask = (row_cols == i)
            diag = row_vals[diag_mask].sum().item() if diag_mask.any() else 0.0
            row_sum = row_vals.sum().item()
            bound = diag + (row_sum - diag)
            if bound > max_bound:
                max_bound = bound
        return float(max_bound)
    elif W.layout == torch.sparse_coo:
        Wcsr = W.to_sparse_csr()
        return gershgorin_upper_bound(Wcsr)
    else:
        absW = W.abs()
        diag = absW.diag()
        row_sum = absW.sum(dim=1)
        bound = torch.max(diag + (row_sum - diag))
        return float(bound.item())

@torch.no_grad()
def max_row_sum_norm(W: torch.Tensor) -> float:
    if W.layout == torch.sparse_csr:
        crow = W.crow_indices()
        val = W.values().abs()
        max_sum = 0.0
        for i in range(W.shape[0]):
            start = int(crow[i].item())
            end = int(crow[i+1].item())
            s = val[start:end].sum().item()
            if s > max_sum:
                max_sum = s
        return float(max_sum)
    elif W.layout == torch.sparse_coo:
        return max_row_sum_norm(W.to_sparse_csr())
    else:
        return float(W.abs().sum(dim=1).max().item())

@torch.no_grad()
def degree_stats(W: torch.Tensor) -> Tuple[float, int, int, float]:
    if W.layout == torch.sparse_csr:
        crow = W.crow_indices()
        deg = (crow[1:] - crow[:-1]).to(torch.float64)
        avg = float(deg.mean().item())
        std = float(deg.std(unbiased=False).item())
        mx = int(deg.max().item())
        mn = int(deg.min().item())
        return avg, mx, mn, std
    elif W.layout == torch.sparse_coo:
        Wc = W.coalesce()
        rows = Wc.indices()[0]
        counts = torch.bincount(rows, minlength=W.shape[0]).to(torch.float64)
        avg = float(counts.mean().item())
        std = float(counts.std(unbiased=False).item())
        mx = int(counts.max().item())
        mn = int(counts.min().item())
        return avg, mx, mn, std
    else:
        deg = (W != 0).sum(dim=1).to(torch.float64)
        avg = float(deg.mean().item())
        std = float(deg.std(unbiased=False).item())
        mx = int(deg.max().item())
        mn = int(deg.min().item())
        return avg, mx, mn, std

@torch.no_grad()
def commutator_non_normality_proxy(W: torch.Tensor, samples: int = 8, device: Optional[torch.device] = None) -> float:
    if device is None:
        device = W.device
    n, _ = size_of_sparse(W)
    WT = sptm(W)
    acc = 0.0
    for _ in range(samples):
        z = torch.empty(n, device=device).bernoulli_(0.5).mul_(2).sub_(1)  # Rademacher
        Wz = spmv(W, z)
        WT_W_z = spmv(WT, Wz)
        WTz = spmv(WT, z)
        W_WT_z = spmv(W, WTz)
        Az = WT_W_z - W_WT_z
        acc += float(Az.pow(2).sum().item())
    return float(math.sqrt(acc / max(samples, 1)))

# ---------- Metrics: weight distribution ----------

@torch.no_grad()
def weight_distribution_stats(W: torch.Tensor) -> Dict[str, float]:
    n, _ = size_of_sparse(W)
    N = n * n

    if W.layout == torch.sparse_csr:
        vals = W.values().to(dtype=torch.float64)
    elif W.layout == torch.sparse_coo:
        vals = W.values().to(dtype=torch.float64)
    else:
        vals = W.reshape(-1).to(dtype=torch.float64)

    nnz = int(vals.numel())
    nz_sum1 = vals.sum()
    nz_sum2 = (vals ** 2).sum()
    nz_sum3 = (vals ** 3).sum()
    nz_sum4 = (vals ** 4).sum()

    if N > 0:
        m1 = (nz_sum1 / N).item()
        m2 = (nz_sum2 / N).item()
        m3 = (nz_sum3 / N).item()
        m4 = (nz_sum4 / N).item()
    else:
        m1 = m2 = m3 = m4 = float('nan')

    mu = m1
    var = m2 - mu * mu
    var = max(var, 0.0)
    std = math.sqrt(var) if var > 0 else float('nan')
    mu3 = m3 - 3 * mu * m2 + 2 * (mu ** 3)
    mu4 = m4 - 4 * mu * m3 + 6 * (mu ** 2) * m2 - 3 * (mu ** 4)
    skew_overall = (mu3 / (var ** 1.5)) if var > 0 else float('nan')
    kurt_ex_overall = (mu4 / (var ** 2) - 3.0) if var > 0 else float('nan')

    if nnz > 0 and N > 0:
        pos_overall = (vals > 0).sum().item() / N
        neg_overall = (vals < 0).sum().item() / N
    else:
        pos_overall = neg_overall = 0.0
    zero_overall = 1.0 - pos_overall - neg_overall

    if nnz > 0:
        mean_nz = (nz_sum1 / nnz).item()
        var_nz = (nz_sum2 / nnz - mean_nz ** 2)
        var_nz = max(var_nz, 0.0)
        std_nz = math.sqrt(var_nz) if var_nz > 0 else float('nan')
        mu3_nz = (nz_sum3 / nnz - 3 * mean_nz * (nz_sum2 / nnz) + 2 * mean_nz ** 3)
        mu4_nz = (nz_sum4 / nnz - 4 * mean_nz * (nz_sum3 / nnz) + 6 * mean_nz ** 2 * (nz_sum2 / nnz) - 3 * mean_nz ** 4)
        skew_nz = (mu3_nz / (var_nz ** 1.5)) if var_nz > 0 else float('nan')
        kurt_ex_nz = (mu4_nz / (var_nz ** 2) - 3.0) if var_nz > 0 else float('nan')
        pos_nz = (vals > 0).sum().item() / nnz
        neg_nz = (vals < 0).sum().item() / nnz
    else:
        mean_nz = std_nz = skew_nz = kurt_ex_nz = float('nan')
        pos_nz = neg_nz = float('nan')

    return {
        "w_mean_overall": m1,
        "w_std_overall": std,
        "w_skew_overall": skew_overall,
        "w_kurt_ex_overall": kurt_ex_overall,
        "w_pos_rate_overall": pos_overall,
        "w_neg_rate_overall": neg_overall,
        "w_zero_rate_overall": zero_overall,
        "w_mean_nz": mean_nz,
        "w_std_nz": std_nz,
        "w_skew_nz": skew_nz,
        "w_kurt_ex_nz": kurt_ex_nz,
        "w_pos_rate_nz": pos_nz,
        "w_neg_rate_nz": neg_nz,
    }

# ---------- Records ----------

@dataclass
class Metrics:
    id: str
    seed: int
    final_train_perplexity: float
    final_val_perplexity: float
    msf_ppl: float
    model_path: str
    N: int
    nnz: int
    density: float
    avg_degree: float
    std_degree: float
    max_degree: int
    min_degree: int
    max_row_sum: float
    gershgorin_upper: float
    spec_norm_est: float
    spec_radius_est: float
    commutator_norm_proxy: float
    leak_mean: Optional[float]
    leak_min: Optional[float]
    leak_max: Optional[float]
    w_mean_overall: float
    w_std_overall: float
    w_skew_overall: float
    w_kurt_ex_overall: float
    w_pos_rate_overall: float
    w_neg_rate_overall: float
    w_zero_rate_overall: float
    w_mean_nz: float
    w_std_nz: float
    w_skew_nz: float
    w_kurt_ex_nz: float
    w_pos_rate_nz: float
    w_neg_rate_nz: float
    notes: str

# ---------- Plotting: validation perplexity boxplot ----------

def save_val_ppl_boxplot(csv_path: str, out_png: str, yscale: str = "linear"):
    """
    Load CSV (runs_csv or reservoir_metrics.csv) and save a box-and-whisker plot
    of 'final_val_perplexity' as a PNG. Uses matplotlib only; one chart per figure.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    vals = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            if "final_val_perplexity" in r and r["final_val_perplexity"] not in ("", "nan", "NaN"):
                try:
                    v = float(r["final_val_perplexity"])
                    if math.isfinite(v):
                        vals.append(v)
                except Exception:
                    pass

    if not vals:
        log("[warn] no valid 'final_val_perplexity' to plot.")
        return

    fig = plt.figure(figsize=(6, 4), dpi=160)
    ax = fig.add_subplot(1, 1, 1)
    ax.boxplot(vals, vert=True, showmeans=True)
    ax.set_ylabel("Validation Perplexity")
    ax.set_title("Validation PPL (Boxplot)")
    if yscale in ("log", "log10", "Log", "LOG"):
        ax.set_yscale("log")
        ax.set_ylabel("Validation Perplexity (log scale)")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    log(f"[plot] saved boxplot -> {out_png}")

# ---------- Pipeline ----------

def analyze_checkpoint(row: Dict[str, str], args) -> Optional[Metrics]:
    model_path = row["model_path"]
    if not os.path.exists(model_path):
        log(f"[skip] not found: {model_path}")
        return None
    try:
        ckpt = torch.load(model_path, map_location="cpu")
    except Exception as e:
        log(f"[skip] failed to load {model_path}: {e}")
        return None

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            state = ckpt["model_state_dict"]
        else:
            state = ckpt
    else:
        log(f"[skip] unexpected checkpoint type in {model_path}")
        return None

    device = torch.device(args.device)
    try:
        mats = extract_W_rec_and_leak(state, device=device)
    except Exception as e:
        log(f"[skip] extracting W_rec failed for {model_path}: {e}")
        return None

    W = mats.W_rec
    n, m = size_of_sparse(W)
    if n != m:
        log(f"[skip] W_rec not square in {model_path}: {W.shape}")
        return None

    nnz = nnz_of_sparse(W)
    density = float(nnz) / float(n * n)

    avg_deg, mx_deg, mn_deg, std_deg = degree_stats(W)
    max_row = max_row_sum_norm(W)
    gbound = gershgorin_upper_bound(W)

    spec_norm = estimate_spectral_norm(W, iters=args.power_iters, device=device)
    spec_rad = estimate_spectral_radius_power(W, iters=args.power_iters, device=device)

    comm_norm = commutator_non_normality_proxy(W, samples=args.hutch_samples, device=device)

    leak_mean = leak_min = leak_max = None
    if mats.leak is not None:
        leak_mean = float(mats.leak.mean().item())
        leak_min  = float(mats.leak.min().item())
        leak_max  = float(mats.leak.max().item())

    wstats = weight_distribution_stats(W)

    notes = ""
    return Metrics(
        id=row.get("id", ""),
        seed=int(float(row.get("seed", "0"))),
        final_train_perplexity=float(row.get("final_train_perplexity", "nan")),
        final_val_perplexity=float(row.get("final_val_perplexity", "nan")),
        msf_ppl=float(row.get("msf_ppl", "nan")),
        model_path=model_path,
        N=n,
        nnz=nnz,
        density=density,
        avg_degree=avg_deg,
        std_degree=std_deg,
        max_degree=mx_deg,
        min_degree=mn_deg,
        max_row_sum=max_row,
        gershgorin_upper=gbound,
        spec_norm_est=spec_norm,
        spec_radius_est=spec_rad,
        commutator_norm_proxy=comm_norm,
        leak_mean=leak_mean,
        leak_min=leak_min,
        leak_max=leak_max,
        w_mean_overall=wstats["w_mean_overall"],
        w_std_overall=wstats["w_std_overall"],
        w_skew_overall=wstats["w_skew_overall"],
        w_kurt_ex_overall=wstats["w_kurt_ex_overall"],
        w_pos_rate_overall=wstats["w_pos_rate_overall"],
        w_neg_rate_overall=wstats["w_neg_rate_overall"],
        w_zero_rate_overall=wstats["w_zero_rate_overall"],
        w_mean_nz=wstats["w_mean_nz"],
        w_std_nz=wstats["w_std_nz"],
        w_skew_nz=wstats["w_skew_nz"],
        w_kurt_ex_nz=wstats["w_kurt_ex_nz"],
        w_pos_rate_nz=wstats["w_pos_rate_nz"],
        w_neg_rate_nz=wstats["w_neg_rate_nz"],
        notes=notes
    )

def read_runs_csv(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)
    return rows

def write_metrics_csv(path: str, records: List[Metrics]):
    fieldnames = list(asdict(records[0]).keys()) if records else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for rec in records:
            wr.writerow(asdict(rec))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_csv", type=str, default="run_table.csv", help="id, seed, PPL, model_path を含む CSV")
    parser.add_argument("--out_csv", type=str, default="reservoir_metrics_mlo.csv")
    parser.add_argument("--device", type=str, default="cpu", help="cpu / cuda")
    parser.add_argument("--max_models", type=int, default=0, help="0 なら全件、>0 なら先頭から制限数のみ解析")
    parser.add_argument("--power_iters", type=int, default=200000, help="パワーイテレーション反復回数（スペクトルノルム/半径推定）")
    parser.add_argument("--hutch_samples", type=int, default=32, help="非正規性プロキシのハッチンソン試行回数")
    parser.add_argument("--boxplot_out", type=str, default="", help="Validation PPL 箱ひげ図の保存先 PNG（空なら作図しない）")
    parser.add_argument("--boxplot_from", type=str, default="", help="箱ひげ図の入力CSV（未指定なら out_csv または runs_csv）")
    parser.add_argument("--boxplot_yscale", type=str, default="linear", choices=["linear", "log"], help="y軸スケール")
    args = parser.parse_args()

    rows = read_runs_csv(args.runs_csv)
    if args.max_models > 0:
        rows = rows[:args.max_models]
    results: List[Metrics] = []
    for i, row in enumerate(rows, 1):
        log(f"[{i}/{len(rows)}] analyzing {row.get('id','')} ...")
        m = analyze_checkpoint(row, args)
        if m is not None:
            results.append(m)

    if results:
        write_metrics_csv(args.out_csv, results)
        log(f"done. wrote: {args.out_csv} ({len(results)} rows)")
    else:
        log("no records written (all skipped or failed).")

    # Optional: make a boxplot of final_val_perplexity
    if args.boxplot_out:
        src_csv = args.boxplot_from if args.boxplot_from else (args.out_csv if results else args.runs_csv)
        try:
            save_val_ppl_boxplot(src_csv, args.boxplot_out, yscale=args.boxplot_yscale)
        except Exception as e:
            log(f"[warn] failed to save boxplot: {e}")

if __name__ == "__main__":
    main()
