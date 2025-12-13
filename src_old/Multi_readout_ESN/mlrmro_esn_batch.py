#!/usr/bin/env python3
import argparse
import os
import math
import datetime
import time
import json
from pathlib import Path
from itertools import islice

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import torch.distributed as dist

from tqdm.auto import tqdm
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaTokenizer, get_cosine_schedule_with_warmup
from huggingface_hub import HfApi, upload_folder
import random
try:
    import deepspeed
    _HAS_DS = True
except ImportError:
    _HAS_DS = False

# ---------- Constants ----------
MAGIC_NUMBER = 20240520
HEADER_INT32 = 256  # 1 KiB header = 256 * int32
VERSION = 1
HEADER_U16 = HEADER_INT32 * 2

# =============================================================================
# Dataset helpers
# =============================================================================
class BinShardsDataset(IterableDataset):
    """Load pre‑tokenised uint16 shards with zero‑copy mmap (rank & worker sharded)."""

    def __init__(self, shard_dir: str | Path, seq_len: int):
        self.seq_len = seq_len
        self.files = sorted(Path(shard_dir).glob("shard_*.bin"))
        if not self.files:
            raise FileNotFoundError(f"no shard_*.bin in {shard_dir}")
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

    # ------------------------------------------------------------------ utils
    def _load_uint16_tokens(self, file: Path) -> torch.Tensor:
        header = torch.from_file(str(file), dtype=torch.int32, size=HEADER_INT32)
        if header[0].item() != MAGIC_NUMBER or header[1].item() != VERSION:
            raise ValueError(f"bad header in {file}")
        num_tok = int(header[2].item())
        tot_u16 = HEADER_U16 + num_tok
        mapped = torch.from_file(str(file), dtype=torch.uint16, size=tot_u16)
        return mapped[HEADER_U16:]

    def _yield_file_tokens(self, file: Path):
        toks = self._load_uint16_tokens(file)
        for j in range(0, len(toks) - self.seq_len + 1, self.seq_len):
            yield toks[j:j + self.seq_len].long()

    # ----------------------------------------------------------------- iter
    def __iter__(self):
        # worker sharding
        worker = get_worker_info()
        worker_id, num_workers = (worker.id, worker.num_workers) if worker else (0, 1)

        for i, f in enumerate(self.files):
            # shard across ranks & workers  → (rank * num_workers + worker_id)
            global_idx = self.rank * num_workers + worker_id
            global_stride = self.world_size * num_workers
            if i % global_stride != global_idx:
                continue
            yield from self._yield_file_tokens(f)

# ---------------------------------------------------------------------------
class StreamingDataset(IterableDataset):
    def __init__(self, dataset_path, split, tokenizer, seq_len):
        self.ds = load_dataset(dataset_path, split=split, streaming=True).shuffle(buffer_size=10, seed=42) #10_000_000
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

    def __iter__(self):
        worker = get_worker_info()
        worker_id, num_workers = (worker.id, worker.num_workers) if worker else (0, 1)
        buf = []
        for idx, ex in enumerate(self.ds):
            global_idx = self.rank * num_workers + worker_id
            global_stride = self.world_size * num_workers
            if idx % global_stride != global_idx:
                continue
            toks = self.tokenizer(ex["text"], return_attention_mask=False)["input_ids"]
            buf.extend(toks)
            while len(buf) >= self.seq_len:
                yield torch.tensor(buf[: self.seq_len], dtype=torch.long)
                buf = buf[self.seq_len:]

# =============================================================================
# misc helpers
# =============================================================================

def load_api_keys(path):
    """Read key=value pairs from api.txt and return dict."""
    keys = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                keys[k.strip()] = v.strip()
    return keys


def prepare_hf_repo_id(write_token, explicit_repo, default_prefix="ESN_MRO_W01_1BT"):
    api = HfApi(token=write_token)
    if explicit_repo:
        return explicit_repo, api
    owner = api.whoami()["name"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    repo_id = f"{owner}/{default_prefix}-{timestamp}"
    return repo_id, api


# =============================================================================
# ESN_mlr network
# =============================================================================

# =============================================================================
# ESN_mlr network (A共有＋前段混合＋Top-k最適化)
# =============================================================================
# =============================================================================
# ESN_mlr_mro (FAST): sparse→dense 置換 + Embedding 入力 + A共有 + Top-k計算削減
# =============================================================================
class ESN_mlr_mro(nn.Module):
    """
    Echo-State Network 言語モデル (multi-leak, multi-readout, low-rank out, fast path).
    - 入力:  one-hot×疎W_in  →  Embedding lookup（dense, 再現値）
    - 再帰:  疎W_rec         →  dense matmul（Tensor Core 利用）
    - 読出:  K個の低ランクB_iを r次元で混合 → 共有Aで1回だけ語彙へ投影
    - router_topk>0 なら未選抜専門家の B_i(h) を計算しない
    """

    def __init__(
        self,
        vocab_size: int,
        reservoir_size: int = 4096,
        d: int = 32,
        spectral_radius: float = 0.99,
        sigma_in: float = 1.0,
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
        activation: str = "tanh",
        dropout: float = 0.1,
        r_out: int = 512,
        num_readouts: int = 4,
        gate_temp: float = 1.0,
        router_entropy: float = 0.0,
        router_balance: float = 0.0,
        router_topk: int = 0,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        assert gate_temp > 0
        self.vocab_size = vocab_size
        self.reservoir_size = reservoir_size
        self.d = d
        self.gamma = d / self.reservoir_size
        self.device = device
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.sigma_in = sigma_in
        self.spectral_radius = spectral_radius
        self.activation = activation
        self.dropout = dropout
        self.r_out = r_out
        self.num_readouts = max(1, int(num_readouts))
        self.gate_temp = gate_temp
        self.router_entropy = router_entropy
        self.router_balance = router_balance
        self.router_topk = router_topk

        # -------- learnable per-neuron leak a_i in [alpha_min, alpha_max]
        a0 = torch.empty(self.reservoir_size, device=device).uniform_(alpha_min, alpha_max)
        eps = 1e-6
        a0_bar = ((a0 - alpha_min) / max(alpha_max - alpha_min, eps)).clamp_(eps, 1 - eps)
        self.param_a = nn.Parameter(torch.log(a0_bar / (1 - a0_bar)))  # unconstrained

        # -------- Build original sparse W_in / W_rec just for init
        W_in_sparse = self._rand_sparse((self.vocab_size, self.reservoir_size), self.gamma, self.sigma_in, device)
        W_rec_sparse = self._rand_sparse((self.reservoir_size, self.reservoir_size), self.gamma, 1.0, device)
        W_rec_sparse = self._scale_spectral_radius(W_rec_sparse, spectral_radius)

        # -------- FAST PATH: densify
        # 入力: Embedding(V, N) に疎行列をそのまま密化コピー（固定＝学習しない）
        self.E_in = nn.Embedding(self.vocab_size, self.reservoir_size)  # (V, N)
        with torch.no_grad():
            self.E_in.weight.copy_(W_in_sparse.to_dense())
        self.E_in.weight.requires_grad_(False)

        # 再帰: dense W_rec^T を buffer で保持（h @ W_rec_T）
        W_rec_dense_T = W_rec_sparse.to_dense().t().contiguous()  # (N, N)
        self.register_buffer("W_rec_T", W_rec_dense_T)

        # -------- K low-rank B_i + shared A
        self.B_list = nn.ModuleList([nn.Linear(self.reservoir_size, r_out, bias=False)
                                     for _ in range(self.num_readouts)])
        self.A = nn.Linear(r_out, self.vocab_size, bias=True)
        self.drop = nn.Dropout(dropout)

        # -------- gating
        self.gate = nn.Linear(self.reservoir_size, self.num_readouts) if self.num_readouts > 1 else None

        # -------- activation
        if activation == "tanh":
            self.act = torch.tanh
        elif activation == "relu":
            self.act = F.relu
        elif activation == "gelu":
            self.act = F.gelu
        else:
            raise ValueError("activation must be tanh / relu / gelu")

        self.register_buffer("h0", torch.zeros(self.reservoir_size, device=device))
        self.last_router_usage = None

        print(
            f"[FAST] ESN_mlr_mro: V={self.vocab_size}, N={self.reservoir_size}, "
            f"gamma={self.gamma:.6f}, act={self.activation}, r_out={self.r_out}, K={self.num_readouts} "
            f"(dense rec / embed in / A-shared / top-k compute)"
        )

    # ----- utils -------------------------------------------------------------
    @staticmethod
    def _rand_sparse(shape, density, scale, device):
        rows, cols = shape
        nnz = int(round(rows * cols * density))
        row_idx = torch.randint(rows, (nnz,), device=device)
        col_idx = torch.randint(cols, (nnz,), device=device)
        vals = torch.randn(nnz, device=device) * scale
        idx = torch.stack([row_idx, col_idx])
        return torch.sparse_coo_tensor(idx, vals, shape, device=device)

    @staticmethod
    @torch.no_grad()
    def _scale_spectral_radius(mat, target_rho, iters=50000):
        v = torch.randn(mat.size(0), 1, device=mat.device)
        v /= v.norm() + 1e-9
        for _ in range(iters):
            v = torch.sparse.mm(mat, v)
            v /= v.norm() + 1e-9
        cur_rho = torch.dot(v.squeeze(), torch.sparse.mm(mat, v).squeeze()).abs()
        mat = mat.coalesce()
        new_vals = mat.values() * (target_rho / (cur_rho + 1e-9))
        return torch.sparse_coo_tensor(mat.indices(), new_vals, mat.size(), device=mat.device)

    def _leak(self):
        a_bar = torch.sigmoid(self.param_a)
        return self.alpha_min + (self.alpha_max - self.alpha_min) * a_bar

    # ----- forward -----------------------------------------------------------
    def forward(self, x: torch.LongTensor, labels: torch.LongTensor | None = None):
        """
        x: (B, T)
        """
        B, T = x.shape
        h = self.h0.expand(B, -1)                  # (B, N)
        a = self._leak().unsqueeze(0)              # (1, N)

        outs = []
        p_hist = []

        for t in range(T):
            # 入力投影: one-hot×W_in → Embedding lookup（超高速）
            u_proj = self.E_in(x[:, t])            # (B, N) float32
            u_proj = u_proj.to(h.dtype)            # bf16 へ

            # 再帰項: h @ W_rec_T（dense matmul, Tensor Core 利用）
            rec = h @ self.W_rec_T                 # (B, N) same dtype as h

            # 状態更新
            pre = (u_proj + rec).clamp_(-10.0, 10.0)
            h = (1 - a) * h + a * self.act(pre)   # (B, N)

            # 読出し（A共有、r次元で混合）
            if self.num_readouts == 1:
                r_mix = self.drop(self.B_list[0](h))      # (B, r)
                logits_t = self.A(r_mix)                  # (B, V)
            else:
                gate_logits = self.gate(h) / self.gate_temp   # (B, K)
                if self.router_topk and self.router_topk < self.num_readouts:
                    topk_val, topk_idx = torch.topk(gate_logits, self.router_topk, dim=-1)
                    mask = torch.full_like(gate_logits, float('-inf'))
                    gate_logits = mask.scatter(1, topk_idx, topk_val)
                p = F.softmax(gate_logits, dim=-1)             # (B, K)
                p_hist.append(p.detach())

                if self.router_topk and self.router_topk < self.num_readouts:
                    with torch.no_grad():
                        used = torch.unique(topk_idx)          # (K_used,)
                    r_used = []
                    for i in used.tolist():
                        r_i = self.B_list[i](h)                # (B, r)
                        r_used.append(r_i.unsqueeze(1))
                    r_used = torch.cat(r_used, dim=1)          # (B, K_used, r)
                    p_used = p.index_select(1, used)           # (B, K_used)
                    r_mix = torch.bmm(p_used.unsqueeze(1).to(r_used.dtype), r_used).squeeze(1)  # (B, r)
                else:
                    r_all = torch.stack([self.B_list[i](h) for i in range(self.num_readouts)], dim=1)  # (B, K, r)
                    r_mix = torch.bmm(p.unsqueeze(1).to(r_all.dtype), r_all).squeeze(1)                # (B, r)

                logits_t = self.A(self.drop(r_mix))            # (B, V)

            outs.append(logits_t)

        logits = torch.stack(outs, dim=1)  # (B, T, V)

        if labels is None:
            if p_hist:
                self.last_router_usage = torch.stack(p_hist, dim=1).mean(dim=(0, 1))  # (K,)
            return logits

        # 言語モデリング損失
        ce = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, self.vocab_size),
            labels[:, 1:].reshape(-1),
        )
        loss = ce

        # ルータ正則化（任意）
        if self.num_readouts > 1 and (self.router_entropy > 0 or self.router_balance > 0):
            P = torch.stack(p_hist, dim=1) if p_hist else None  # (B, T, K)
            if P is not None:
                if self.router_entropy > 0:
                    ent = (-P * (P + 1e-8).log()).sum(dim=-1).mean()
                    loss = loss + self.router_entropy * ent
                if self.router_balance > 0:
                    mean_p = P.mean(dim=(0, 1))
                    target = torch.full_like(mean_p, 1.0 / self.num_readouts)
                    bal = F.mse_loss(mean_p, target)
                    loss = loss + self.router_balance * bal
                self.last_router_usage = P.mean(dim=(0, 1)).detach()

        return logits, loss
# class ESN_mlr_mro(nn.Module):
#     """
#     Echo-State Network 言語モデル (multi-leak, sparse, multi-readout, low-rank out).
#     - 入力/再帰は固定スパース (ESN)
#     - leak rate a_i は学習可能（sigmoid で [alpha_min, alpha_max] へ拘束）
#     - 読み出しは K 個の低ランクヘッドを mixture（ソフトルータ）で加重
#     """

#     def __init__(
#         self,
#         vocab_size: int,
#         reservoir_size: int = 4096,
#         d: int = 32,
#         spectral_radius: float = 0.99,
#         sigma_in: float = 1.0,
#         alpha_min: float = 0.0,
#         alpha_max: float = 1.0,
#         activation: str = "tanh",
#         dropout: float = 0.1,
#         r_out: int = 512,
#         num_readouts: int = 4,
#         gate_temp: float = 1.0,
#         router_entropy: float = 0.0,
#         router_balance: float = 0.0,
#         router_topk: int = 0,
#         device: torch.device = torch.device("cpu"),
#     ):
#         super().__init__()
#         assert gate_temp > 0, "gate_temp must be > 0"
#         self.vocab_size = vocab_size
#         self.reservoir_size = reservoir_size
#         self.d = d
#         self.gamma = d / self.reservoir_size      # connectivity
#         self.device = device
#         self.alpha_min = alpha_min
#         self.alpha_max = alpha_max
#         self.sigma_in = sigma_in
#         self.spectral_radius = spectral_radius
#         self.activation = activation
#         self.dropout = dropout
#         self.r_out = r_out
#         self.num_readouts = max(1, int(num_readouts))
#         self.gate_temp = gate_temp
#         self.router_entropy = router_entropy
#         self.router_balance = router_balance
#         self.router_topk = router_topk

#         # ---------- learnable leak rate a_i in [alpha_min, alpha_max] ----------
#         # 逆シグモイド初期化（[alpha_min, alpha_max] 一様にサンプル→逆写像）
#         a0 = torch.empty(self.reservoir_size, device=device).uniform_(alpha_min, alpha_max)
#         eps = 1e-6
#         a0_bar = (a0 - alpha_min) / max(alpha_max - alpha_min, eps)
#         a0_bar = a0_bar.clamp_(eps, 1 - eps)
#         self.param_a = nn.Parameter(torch.log(a0_bar / (1 - a0_bar)))  # unconstrained

#         # ---------- sparse W_in (V × N) ----------
#         W_in = self._rand_sparse((self.vocab_size, self.reservoir_size), self.gamma, self.sigma_in, device)
#         self.register_buffer("W_in_T", W_in.transpose(0, 1).coalesce())  # (N,V)

#         # ---------- sparse W_rec (N × N) ----------
#         W_rec = self._rand_sparse((self.reservoir_size, self.reservoir_size), self.gamma, 1.0, device)
#         W_rec = self._scale_spectral_radius(W_rec, spectral_radius)
#         self.register_buffer("W_rec", W_rec.coalesce())

#         # ---------- K low-rank readouts ----------
#         self.B_list = nn.ModuleList([nn.Linear(self.reservoir_size, r_out, bias=False) for _ in range(self.num_readouts)])
#         self.A_list = nn.ModuleList([nn.Linear(r_out, self.vocab_size, bias=True)      for _ in range(self.num_readouts)])
#         self.drop = nn.Dropout(dropout)

#         # ---------- gating (soft router) ----------
#         self.gate = nn.Linear(self.reservoir_size, self.num_readouts) if self.num_readouts > 1 else None

#         # ---------- activation ----------
#         if activation == "tanh":
#             self.act = torch.tanh
#         elif activation == "relu":
#             self.act = F.relu
#         elif activation == "gelu":
#             self.act = F.gelu
#         else:
#             raise ValueError("activation must be tanh / relu / gelu")

#         self.register_buffer("h0", torch.zeros(self.reservoir_size, device=device))

#         print(
#             f"ESN_mlr (MR) initialized: V={self.vocab_size}, N={self.reservoir_size}, d={self.d}, "
#             f"gamma={self.gamma:.6f}, sigma_in={self.sigma_in}, a∈[{self.alpha_min},{self.alpha_max}], "
#             f"act={self.activation}, drop={self.dropout}, r_out={self.r_out}, K={self.num_readouts}"
#         )

#         # for optional logging
#         self.last_router_usage = None  # shape (K,)

#     # ======================================================================
#     # utils
#     # ----------------------------------------------------------------------
#     @staticmethod
#     def _rand_sparse(shape, density, scale, device):
#         rows, cols = shape
#         nnz = int(round(rows * cols * density))
#         row_idx = torch.randint(rows, (nnz,), device=device)
#         col_idx = torch.randint(cols, (nnz,), device=device)
#         vals = torch.randn(nnz, device=device) * scale
#         idx = torch.stack([row_idx, col_idx])
#         return torch.sparse_coo_tensor(idx, vals, shape, device=device)

#     @staticmethod
#     @torch.no_grad()
#     def _scale_spectral_radius(mat, target_rho, iters=50000):
#         v = torch.randn(mat.size(0), 1, device=mat.device)
#         v /= v.norm() + 1e-9
#         for _ in range(iters):
#             v = torch.sparse.mm(mat, v)
#             v /= v.norm() + 1e-9
#         cur_rho = torch.dot(v.squeeze(), torch.sparse.mm(mat, v).squeeze()).abs()
#         mat = mat.coalesce()
#         new_vals = mat.values() * (target_rho / (cur_rho + 1e-9))
#         return torch.sparse_coo_tensor(mat.indices(), new_vals, mat.size(), device=mat.device)

#     def _leak(self):
#         # sigmoid で [alpha_min, alpha_max] に拘束
#         a_bar = torch.sigmoid(self.param_a)
#         return self.alpha_min + (self.alpha_max - self.alpha_min) * a_bar

#     # ======================================================================
#     # forward
#     # ----------------------------------------------------------------------
#     def forward(self, x: torch.LongTensor, labels: torch.LongTensor | None = None):
#         """
#         x: (B, T)
#         """
#         B, T = x.shape
#         h = self.h0.expand(B, -1)            # (B, N)
#         a = self._leak().unsqueeze(0)        # (1, N)

#         outs = []
#         p_hist = []  # for regularization & stats

#         for t in range(T):
#             # ----- one-hot → sparse matmul（f32で安定化） -------------------------
#             one_hot = F.one_hot(x[:, t], num_classes=self.vocab_size).to(h.dtype)
#             with torch.amp.autocast(device_type="cuda", enabled=False):
#                 one_hot_f32 = one_hot.t().float()  # (V,B)
#                 u_proj = torch.sparse.mm(self.W_in_T.float(), one_hot_f32).t()   # (B,N) f32
#                 rec    = torch.sparse.mm(self.W_rec.float(),   h.float().t()).t() # (B,N) f32
#             u_proj = u_proj.to(h.dtype)
#             rec    = rec.to(h.dtype)

#             # ----- state update -----------------------------------------------
#             pre = (u_proj + rec).clamp_(-10.0, 10.0)
#             h = (1 - a) * h + a * self.act(pre)   # (B,N)

#             # ----- gating & readouts ------------------------------------------
#             if self.num_readouts == 1:
#                 r = self.drop(self.B_list[0](h))
#                 logits_t = self.A_list[0](r)
#             else:
#                 # gate in float32 for numerical stability
#                 gate_logits = self.gate(h.float()) / self.gate_temp  # (B,K)
#                 if self.router_topk and self.router_topk < self.num_readouts:
#                     # Top-k sparse softmax（ソフト版）
#                     topk_val, topk_idx = torch.topk(gate_logits, self.router_topk, dim=-1)
#                     mask = torch.full_like(gate_logits, fill_value=float('-inf'))
#                     gate_logits = mask.scatter(1, topk_idx, topk_val)
#                 p = F.softmax(gate_logits, dim=-1)  # (B,K)
#                 p_hist.append(p.detach())

#                 # 各エキスパートの出力
#                 exps = []
#                 for i in range(self.num_readouts):
#                     r = self.drop(self.B_list[i](h))         # (B, r_out)
#                     y = self.A_list[i](r)                    # (B, V)
#                     exps.append(y)
#                 exps = torch.stack(exps, dim=1)              # (B, K, V)
#                 logits_t = (p.to(exps.dtype).unsqueeze(-1) * exps).sum(dim=1)  # (B,V)

#             outs.append(logits_t)

#         logits = torch.stack(outs, dim=1)  # (B, T, V)

#         if labels is None:
#             # サンプリング用（互換性維持）
#             if p_hist:
#                 self.last_router_usage = torch.stack(p_hist, dim=1).mean(dim=(0,1))  # (K,)
#             return logits

#         # --- 言語モデリング損失 ---
#         ce = F.cross_entropy(
#             logits[:, :-1, :].reshape(-1, self.vocab_size),
#             labels[:, 1:].reshape(-1),
#         )
#         loss = ce

#         # --- ルータ正則化（任意） ---
#         if self.num_readouts > 1 and (self.router_entropy > 0 or self.router_balance > 0):
#             P = torch.stack(p_hist, dim=1) if p_hist else None  # (B,T,K)
#             if P is not None:
#                 if self.router_entropy > 0:
#                     ent = (-P * (P + 1e-8).log()).sum(dim=-1).mean()  # 平均エントロピー
#                     loss = loss + self.router_entropy * ent
#                 if self.router_balance > 0:
#                     mean_p = P.mean(dim=(0,1))                       # (K,)
#                     target = torch.full_like(mean_p, 1.0 / self.num_readouts)
#                     bal = F.mse_loss(mean_p, target)
#                     loss = loss + self.router_balance * bal
#                 # ログ用
#                 self.last_router_usage = P.mean(dim=(0,1)).detach()

#         return logits, loss
# =============================================================================
# Sampling util (unchanged)
# =============================================================================
@torch.no_grad()
def sample_sequence(
    model,
    input_ids: torch.LongTensor,
    max_new_tokens: int = 20,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    min_p: float = 0.0,
    device: torch.device | None = None,
):
    device = device or next(model.parameters()).device
    generated = input_ids.to(device)
    for _ in range(max_new_tokens):
        outputs = model(generated)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        next_logits = logits[:, -1, :] / max(temperature, 1e-8)
        if top_k > 0:
            values, _ = torch.topk(next_logits, top_k)
            min_value = values[:, -1].unsqueeze(-1)
            next_logits = torch.where(next_logits < min_value, torch.full_like(next_logits, float('-inf')), next_logits)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumulative_probs > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_logits[mask] = float('-inf')
            next_logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)
        if min_p > 0.0:
            probs = F.softmax(next_logits, dim=-1)
            next_logits = torch.where(probs < min_p, torch.full_like(next_logits, float('-inf')), next_logits)
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)
    return generated

# =============================================================================
# Data loader helper
# =============================================================================

def get_train_loader(tokenizer_path: str, path: str, seq_len: int, batch_size: int):
    if Path(path).is_dir():
        ds = BinShardsDataset(path, seq_len)
        print(f"[data] BinShardsDataset with {len(list(Path(path).glob('shard_*.bin')))} files")
        tot = len(list(Path(path).glob('shard_*.bin')))
        per_rank = (tot + ds.world_size - 1) // ds.world_size
        print(f"[data] BinShardsDataset: {tot} shards → {per_rank} / rank (world={ds.world_size})")

    else:
        tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        ds = StreamingDataset(path, "train", tok, seq_len)  # original tokenizer‑on‑the‑fly dataset
        print("[data] HF streaming dataset", path)
    return DataLoader(ds, batch_size=batch_size, num_workers=2 if Path(path).is_dir() else 6,
                      pin_memory=True, drop_last=True)
def get_validation_blocks(hf_dataset, tokenizer, seq_len, max_blocks=100):
    blocks = []
    buffer = []
    for sample in hf_dataset:
        text = sample.get("text", "")
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        buffer.extend(token_ids)
        while len(buffer) >= seq_len and len(blocks) < max_blocks:
            block = buffer[:seq_len]
            blocks.append({
                "input_ids": torch.tensor(block, dtype=torch.long),
                "labels": torch.tensor(block, dtype=torch.long)
            })
            buffer = buffer[seq_len:]
        if len(blocks) >= max_blocks:
            break
    return blocks

def safe_decode(token_list, tokenizer):
    try:
        return tokenizer.decode(token_list, skip_special_tokens=True)
    except Exception:
        s = "".join([chr(max(32, t)) for t in token_list])
        return s.encode("utf-8", "replace").decode("utf-8")
# =============================================================================
# Argument parser
# =============================================================================

def parse_args_esn():
    parser = argparse.ArgumentParser()
    # dist / DS
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)), help="Local rank for distributed training")
    parser.add_argument("--deepspeed_config", type=str, default="ds_config.json", help="Path to DeepSpeed config file")
    parser.add_argument("--use_deepspeed", action="store_true", help="Enable DeepSpeed if available and GPUs > 1")
    # training hyperparameters
    parser.add_argument("--local_batch_size", type=int, default=128, help="Micro batch size per GPU")
    parser.add_argument("--use_gpu_amount", type=int, default=max(torch.cuda.device_count(), 1), help="Total number of GPUs for global batch calculation, if use cpu, set to 1")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--validate_every_steps", type=int, default=250, help="Validate every N steps")
    parser.add_argument("--save_checkpoint_every_steps", type=int, default=2000, help="Save checkpoint every N steps")
    parser.add_argument("--generate_every", type=int, default=1000, help="Generate every N steps")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--total_tokens", type=float, default=10e8, help="Total number of tokens to train on")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs (streaming ignored)")
    parser.add_argument("--grad_clip_norm", type=float, default=0, help="Global norm for gradient clipping (<=0 to disable)")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for AdamW")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 for AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay for AdamW")
    parser.add_argument("--tokenizer_path", type=str, default="meta-llama/Llama-2-7b-hf", help="Tokenizer path")

    # ESN
    parser.add_argument("--reservoir_size", type=int, default=2048, help="ESN reservoir size (N_rec)")
    parser.add_argument("--d", type=int, default=32, help="Reservoir sparsity (d/N_rec)")
    parser.add_argument("--spectral_radius", type=float, default=0.99, help="Reservoir spectral radius")
    parser.add_argument("--sigma_in", type=float, default=1.0, help="Input weight init scale")
    parser.add_argument("--alpha_min", type=float, default=0.0, help="Leak rate a_i min")
    parser.add_argument("--alpha_max", type=float, default=1.0, help="Leak rate a_i max")
    parser.add_argument("--activation", choices=["relu","tanh","gelu"], default="tanh", help="Reservoir activation")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout before output")
    parser.add_argument("--r_out", type=int, default=512, help="Output size")
    parser.add_argument("--num_readouts", type=int, default=4, help="Number of readout experts (K). 1 = disable multi-readout")
    parser.add_argument("--gate_temp", type=float, default=1.0, help="Softmax temperature for gating (>0)")
    parser.add_argument("--router_entropy", type=float, default=0.0, help="Weight for router entropy regularization (encourage confident routing if >0)")
    parser.add_argument("--router_balance", type=float, default=0.0, help="Weight for router load-balancing regularization")
    parser.add_argument("--router_topk", type=int, default=0, help="If >0, keep top-k experts per step (dense weighted sum otherwise)")

    # Dataset vesteinn/babylm
    parser.add_argument("--dataset_path", type=str, default="HuggingFaceFW/fineweb-edu", help="Dataset path for Training")
    parser.add_argument("--val_dataset_path", type=str, default="vesteinn/babylm", help="Dataset path for Validation")
    # Weights & Biases
    parser.add_argument("--wandb_project", type=str, default="ESN Lanugage Model mlo1B", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name (default: generated by WandB)")
    parser.add_argument("--api_file", type=str, default="api.txt", help="API file path")

    parser.add_argument("--hf_repo", type=str, default=None, help="Upload destination like 'username/gpt2-scratch-64M'. If None, skip upload")
    parser.add_argument("--hf_private", action="store_false", help="Create HF repo as private")


    # JIT compile
    parser.add_argument("--enable_compile", action="store_true", help="Enable torch.compile (unstable with sparse)")

    return parser.parse_args()

# =============================================================================
# main
# =============================================================================

def ESN_experiment(lr,num_readouts,reservoir_size):
    # =============================================================================
    # Evaluation helpers (mean‑so‑far PPL)
    # =============================================================================
    @torch.no_grad()
    def compute_mean_so_far_ppl(model, blocks, device, ks):
        results = {k: [] for k in ks}
        for block in tqdm(blocks, desc="Computing mean‑so‑far PPL"):
            ids = block["input_ids"].to(device).unsqueeze(0)
            logits = model(ids)
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            true_lp = torch.gather(log_probs, 2, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            cum_lp = torch.cumsum(true_lp.squeeze(0), dim=0)  # (T‑1,)
            lengths = torch.arange(1, cum_lp.size(0) + 1, device=device)
            mean_nll = -cum_lp / lengths
            for k in ks:
                idx = min(k, mean_nll.size(0)) - 1
                results[k].append(mean_nll[idx].exp().item())
        return {k: {"mean_ppl": sum(v) / len(v)} for k, v in results.items()}
    id = {datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}
    args = parse_args_esn()
    print(args)
    seed = random.randint(1000000, 5000000)
    #seed = 509971
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import os
    os.makedirs("./reports", exist_ok=True)
    WANDB_AVAILABLE = False
    api_keys = load_api_keys(args.api_file)
    if "WANDB_API_KEY" in api_keys:
        os.environ["WANDB_API_KEY"] = api_keys["WANDB_API_KEY"]
    if "HF_READ_TOKEN" in api_keys:
        from huggingface_hub import login

        login(token=api_keys["HF_WRITE_TOKEN"])
    # distributed setup (only NCCL backend)
    if args.use_deepspeed and _HAS_DS and torch.cuda.device_count() > 1:
        torch.cuda.set_device(args.local_rank)
        distributed = True
    else:
        distributed = False

    world_size = dist.get_world_size() if (distributed and dist.is_initialized()) else 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device is {device}")
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.bos_token = tokenizer.eos_token
    print("Initializing Model")
    # model
    model = ESN_mlr_mro(
        vocab_size=tokenizer.vocab_size,
        reservoir_size=reservoir_size,
        d=args.d,
        spectral_radius=args.spectral_radius,
        sigma_in=args.sigma_in,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        activation=args.activation,
        dropout=args.dropout,
        r_out=args.r_out,
        num_readouts=num_readouts,
        gate_temp=args.gate_temp,
        router_entropy=args.router_entropy,
        router_balance=args.router_balance,
        router_topk=args.router_topk,
        device=device,
    ).to(device).to(torch.bfloat16)
    param_millions = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"parameter count: {param_millions:.2f}M")

    if distributed:
        # Deepspeedを使うときは，deepspeedがoptimizer/ schedulerを内部に持つ
        model, optimizer, _, _ = deepspeed.initialize(
            args=args,
            model=model,
            model_parameters=model.parameters(),
        )
        scheduler = None  # handled by DeepSpeed
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(args.beta1, args.beta2),
            #weight_decay=0.1,
            weight_decay=args.weight_decay,
        )
        total_steps = math.ceil(args.total_tokens / (args.local_batch_size * args.use_gpu_amount * args.seq_len))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=math.ceil(0.1 * total_steps),
            num_training_steps=total_steps,
        )
        model.to(device)
    print("loading train data")
    train_loader = get_train_loader(args.tokenizer_path, args.dataset_path, args.seq_len, args.local_batch_size)


    # validation loader unchanged (SlimPajama slice)
    #val_ds = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True)
    print("loading validation data")
    val_ds = load_dataset(args.val_dataset_path, split="test", streaming=False)
    val_blocks = get_validation_blocks(list(islice(val_ds, 100)), tokenizer, args.seq_len, 1)
    val_loader = DataLoader(val_blocks, batch_size=args.local_batch_size, num_workers=0, pin_memory=True)

    max_steps = math.ceil(args.total_tokens / (args.local_batch_size * args.use_gpu_amount * args.seq_len))

    tokens_seen_local = 0
    tokens_seen_global = 0
    success_flag =True
    val_loss = 10.75

    start_time = time.time()
    max_mem_mb = 0.0
    generated_text_log = []
    model.train()
    if not distributed or args.local_rank == 0:
            wandb_run_name = args.wandb_run_name or f"ESN_mlo ({param_millions:.2f}M N{args.reservoir_size}_mlr{num_readouts}_lr{args.learning_rate}_batch_size{args.local_batch_size}_seq_len{args.seq_len}_sigma_in{args.sigma_in}_spectral_radius{args.spectral_radius}_sparsity{1-(args.d/args.reservoir_size)}_dropout{args.dropout}_r_out{args.r_out}_{id}"
            wandb.init(project=args.wandb_project, name=wandb_run_name, config=vars(args))
            WANDB_AVAILABLE = True

    print("trainging begin")
    for step, batch in enumerate(train_loader, start=1):
        ids = batch.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, loss = model(ids, labels=ids)

        if distributed:
            model.backward(loss)
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.module.parameters(), args.grad_clip_norm)
            model.step()
            #current_lr = optimizer.param_groups[0]["lr"]
            current_lr = model.get_lr()[0]
        else:
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            optimizer.zero_grad()

        step_time = time.time() - start_time
        tokens_seen_local += args.local_batch_size * args.seq_len
        #tokens_seen_global += args.local_batch_size * args.seq_len * dist.get_world_size()
        tokens_seen_global += args.local_batch_size * args.seq_len * world_size
        tokens_per_sec_global = tokens_seen_global / step_time if step_time > 0 else 0.0
        tokens_per_sec_local = tokens_seen_local / step_time if step_time > 0 else 0.0
        vram_mb = torch.cuda.max_memory_allocated() / 1e6
        max_mem_mb = max(max_mem_mb, vram_mb)

        if (not distributed) or args.local_rank == 0:
            wandb.log(
                {
                    "train_loss": loss.item(),
                    "train_perplexity": math.exp(loss.item()),
                    "current_lr": current_lr,
                    "seenedtokens": tokens_seen_local,
                    "seenedtokens_global": tokens_seen_global,
                    "tokens_per_sec": tokens_per_sec_local,
                    "tokens_per_sec_global": tokens_per_sec_global,
                    "max_mem_mb": max_mem_mb,
                    "step": step,
                    "max_steps": max_steps,
                },
                step=step,
            )
        if step % args.validate_every_steps == 0:
            model.eval()
            val_loss_list = []
            with torch.no_grad():
                for v_batch in islice(val_loader, args.validate_every_steps):
                    v_ids = v_batch["input_ids"].to(device)
                    v_labels = v_batch["labels"].to(device)
                    v_logits, v_loss = model(v_ids, labels=v_labels)
                    val_loss_list.append(v_loss.item())
            val_loss = sum(val_loss_list) / len(val_loss_list)
            if (not distributed) or args.local_rank == 0:
                wandb.log(
                    {
                        "val_loss": val_loss,
                        "val_perplexity": math.exp(val_loss),
                    }
                )
            model.train()

        if step % args.save_checkpoint_every_steps == 0 and args.local_rank==0:
            ckpt_name = f"checkpoint_step{step}_tokens{tokens_seen_global}.pt"
            save_dir = f"./checkpoint/ESN_ml ({param_millions:.2f}M N{args.reservoir_size}_nro{num_readouts}_batch_size{args.local_batch_size}_seq_len{args.seq_len}_sigma_in{args.sigma_in}_spectral_radius{args.spectral_radius}_sparsity{1-(args.d/args.reservoir_size)}_dropout{args.dropout}_r_out{args.r_out}_{id}"
            os.makedirs(save_dir, exist_ok=True)
            if distributed:
                model.save_checkpoint(save_dir=f"./{save_dir}/{ckpt_name}", tag=f"step_{step}")
            else:
                torch.save(model.state_dict(), f"./{save_dir}/{ckpt_name}")

        if step % args.generate_every == 0 and ((not distributed) or args.local_rank == 0):
            model.eval()
            for prompt in ["Hello,", "I'm"]:
                inp_ids = tokenizer.encode(prompt, add_special_tokens=False)
                inp = torch.tensor(inp_ids, dtype=torch.long).to(device)
                generated = sample_sequence(
                    model if not distributed else model.module,
                    inp.unsqueeze(0),
                    max_new_tokens=20,
                    temperature=0.8,
                    top_k=40,
                    top_p=0.9,
                    min_p=0.01,
                )
                output_str = safe_decode(generated[0].tolist(), tokenizer)
                wandb.log({"generated": wandb.Html(f"<b>{prompt}</b>{output_str}")})
                generated_text_log.append(output_str)
            model.train()
        # if step >= max_steps:
        #     break
        if step % 500 == 0:
            print(f"[log] train loss: {loss.item()}, step: {step}, tokens_seen_global: {tokens_seen_global}, tokens_per_sec_global: {tokens_per_sec_global}, max_mem_mb: {max_mem_mb}")
        if tokens_seen_global >= args.total_tokens:
            break
        if val_loss > 10.8:
            print(f"Training is stopped because val_loss is too high: {val_loss}")
            success_flag = False
            break
    # ---------- training finished ----------
    total_train_time = time.time() - start_time
    is_master = (not distributed) or args.local_rank == 0
    if is_master:
        final_dir = "hf_upload"
        os.makedirs(final_dir, exist_ok=True)
        torch.save(
            model.module.state_dict() if distributed else model.state_dict(),
            os.path.join(final_dir, "pytorch_model.bin"),
        )
        with open(os.path.join(final_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

        # upload to HF
        if "HF_WRITE_TOKEN" in api_keys and success_flag:
            repo_id, api = prepare_hf_repo_id(api_keys["HF_WRITE_TOKEN"], args.hf_repo, default_prefix=f"ESN_MRO{param_millions:.2f}M_1BT_mlo_N{args.reservoir_size}")
            api.create_repo(repo_id=repo_id, exist_ok=True, private=args.hf_private)
            upload_folder(repo_id=repo_id, folder_path=final_dir, path_in_repo=".", token=api_keys["HF_WRITE_TOKEN"], ignore_patterns=["*.pt"])
            print(f"✅ Model pushed to https://huggingface.co/{repo_id}")
        else:
            print("HF upload skipped (token absent or repo not specified).")

        # stats & report
        param_count = sum(p.numel() for p in model.parameters())
        report = {
            "run_name": wandb.run.name if WANDB_AVAILABLE else "offline_run",
            "hyperparameters": vars(args),
            "parameter_count": param_count,
            "max_gpu_memory_MB": max_mem_mb,
            "training_time_sec": total_train_time,
            "final_train_loss": loss.item(),
            "final_train_perplexity": math.exp(loss.item()),
            "final_val_loss": val_loss if 'val_loss' in locals() else None,
            "final_val_perplexity": math.exp(val_loss) if 'val_loss' in locals() else None,
            "generated": generated_text_log,
            "seed": seed,
        }
    if success_flag:
        # inference memory & speed test
        test_input = torch.randint(0, tokenizer.vocab_size, (1, args.seq_len), device=device)
        torch.cuda.reset_peak_memory_stats()
        t_inf_start = time.time()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _ = model(test_input)
        infer_time = time.time() - t_inf_start
        infer_tok_per_sec = 1024 / infer_time if infer_time > 0 else 0.0
        infer_mem_mb = torch.cuda.max_memory_allocated() / 1e6
        report.update({"inference_tok_per_sec": infer_tok_per_sec, "inference_mem_MB": infer_mem_mb})
        seq_len_test = args.seq_len * 8
        print(f"▶️ babylm テストセットで mean-so-far-PPL を最大 {seq_len_test} トークンまで計測します…")
        test_ds = load_dataset(args.val_dataset_path, split="test", streaming=False)
        buffer = []
        test_blocks = []
        for ex in test_ds:
            toks = tokenizer(ex["text"], return_attention_mask=False)["input_ids"]
            buffer.extend(toks)
            while len(buffer) >= seq_len_test:
                blk = buffer[:seq_len_test]
                test_blocks.append({
                    "input_ids": torch.tensor(blk, dtype=torch.long),
                    "labels":   torch.tensor(blk, dtype=torch.long),
                })
                buffer = buffer[seq_len_test:]
        test_blocks = test_blocks[:100]
        ks = list(range(1, seq_len_test + 1))

        torch.cuda.reset_peak_memory_stats()
        t_inf_start = time.time()
        mean_so_far = compute_mean_so_far_ppl(
            model if not distributed else model.module,
            test_blocks,
            device,
            ks
        )
        t_inf_end = time.time()
        inf_time = t_inf_end - t_inf_start
        total_inf_tokens = len(test_blocks) * seq_len_test
        infer_tok_per_sec = total_inf_tokens / inf_time if inf_time > 0 else 0.0
        infer_mem_mb = torch.cuda.max_memory_allocated() / 1e6
        report.update({
            "inference_tok_per_sec": infer_tok_per_sec,
            "inference_mem_MB": infer_mem_mb
        })
        print(f"Inference time: {inf_time:.2f}s, Tokens/sec: {infer_tok_per_sec:.2f}, Memory: {infer_mem_mb:.2f}MB")
        if WANDB_AVAILABLE:
            wandb.log({"inference_time": inf_time, "inference_tok_per_sec": infer_tok_per_sec, "inference_mem_MB": infer_mem_mb})

        report_ks = []
        k = 1
        while k <= seq_len_test:
            report_ks.append(k)
            k *=2
        report_ks = [k for k in report_ks if k <= seq_len_test]

        if WANDB_AVAILABLE:
            table = wandb.Table(columns = ["token_length", "perplexity"])
            for k in report_ks:
                table.add_data(k,mean_so_far[k]["mean_ppl"])
            chart = wandb.plot.line(
                table,
                x ="token_length",
                y ="perplexity",
                title="Mean-so-far PPL vs Token Length"
            )
            wandb.log({"mean_so_far_ppl_curve": chart})
        # for k in report_ks:
        #     v = mean_so_far[k]
        #     print(f" mean-so-far@{k:4d} →  NLL={v['mean_nll']:.4f},  PPL={v['mean_ppl']:.2f}")
        #     if WANDB_AVAILABLE:
        #         wandb.log({f"test_mean_so_far_ppl@{k}": v["mean_ppl"]})

        # 5) JSON レポートにも一括格納（必要なら全 ks を文字列化して保存）
        report["test_mean_so_far_ppl_curve"] = {
            k: mean_so_far[k]["mean_ppl"] for k in ks
        }

        report_path = f"./reports/{report['run_name']}_report.txt"
        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=2))
        print(f"📄 Training report written to {report_path}")

    if WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    for i in range(3):
        for lr in [1e-3,1e-4,1e-5]:
            for num_readouts in [4,8,16,2]:
                reservoir_size =1024

                try:
                    print(f"lr: {lr}")
                    ESN_experiment(lr,num_readouts,reservoir_size)
                    torch.cuda.empty_cache()
                    time.sleep(10)
                except Exception as e:
                    print(e)
                    with open("./error.txt", "a") as f:
                        f.write(str(e)+"\n")
                    time.sleep(10)
        
