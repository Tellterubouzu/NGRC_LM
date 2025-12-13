




#!/usr/bin/env python3
import argparse
import os
import math
import datetime
import time
import json
from pathlib import Path
from itertools import islice

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

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- ユーティリティ：スペクトル半径スケーリング（近似：パワーイテレーション） ----------
def _scale_to_spectral_radius(W, target_rho=0.99, iters=50):
    # W: (N, N)
    with torch.no_grad():
        v = torch.randn(W.size(0), device=W.device)
        v = v / (v.norm() + 1e-8)
        for _ in range(iters):
            v = W @ v
            v = v / (v.norm() + 1e-8)
        # Rayleigh quotient 近似
        rho = torch.dot(v, W @ v).abs().sqrt().clamp(min=1e-8)
        W.mul_(target_rho / rho)
    return W

@torch.no_grad()
def compute_mean_so_far_ppl(model, blocks, device, ks):
    """
    blocks: [{"input_ids": LongTensor(T), "labels": LongTensor(T)}, ...]
            ※ 各要素は B=1 相当の1系列でもOK（内部で B 次元は付与しない）
    ks:     list[int]  例: [1,2,4,...,Tmax] または 等間隔
    return: {k: {"mean_ppl": float}}
    """
    model.eval()
    out = {k: [] for k in ks}
    for blk in blocks:
        ids = blk["input_ids"].to(device).unsqueeze(0)  # (1, T)
        logits, _ = model(ids, targets=None)            # (1, T, V)
        # 次トークンの対数尤度
        lp = F.log_softmax(logits[:, :-1, :], dim=-1)   # (1, T-1, V)
        gold = ids[:, 1:]                                # (1, T-1)
        true_lp = torch.gather(lp, 2, gold.unsqueeze(-1)).squeeze(-1)  # (1, T-1)
        true_lp = true_lp.squeeze(0)                     # (T-1,)

        cum_lp = torch.cumsum(true_lp, dim=0)            # (T-1,)
        denom = torch.arange(1, cum_lp.numel() + 1, device=ids.device, dtype=cum_lp.dtype)
        mean_nll = -cum_lp / denom                       # (T-1,)

        for k in ks:
            if k <= 0:
                continue
            idx = min(k, mean_nll.numel()) - 1
            out[k].append(mean_nll[idx].exp().item())

    return {k: {"mean_ppl": (sum(v) / max(1, len(v)))} for k, v in out.items()}

# --------- 1ヘッド分のESN（リーク率ベクトル、疎度指定、固定パラメタ） ----------
class LeakyESNHead(nn.Module):
    def __init__(self, d_in, n_state, rho_rec=0.99, sigma_in=1.0,
                 d_connect=32, alpha_min=0.0, alpha_max=1.0, use_tanh=True, device=None):
        super().__init__()
        self.d_in = d_in
        self.n_state = n_state
        self.use_tanh = use_tanh

        dev = device or torch.device("cpu")

        # Win ~ Bernoulli(γ) ⊙ N(0, σ_in^2)
        gamma_in = min(1.0, d_connect / max(1, d_in))
        Win = (torch.rand(n_state, d_in, device=dev) < gamma_in).float()
        Win = Win * torch.randn_like(Win) * sigma_in
        self.register_buffer("Win", Win)

        # Wrec ~ Bernoulli(γ) ⊙ N(0, 1) → スペクトル半径調整
        gamma_rec = min(1.0, d_connect / max(1, n_state))
        Wrec = (torch.rand(n_state, n_state, device=dev) < gamma_rec).float()
        Wrec = Wrec * torch.randn_like(Wrec)
        _scale_to_spectral_radius(Wrec, target_rho=rho_rec)
        self.register_buffer("Wrec", Wrec)

        # a ~ Uniform(αmin, αmax) （ベクトル；多時間尺度）
        a = torch.empty(n_state, device=dev).uniform_(alpha_min, alpha_max)
        self.register_buffer("a", a)

        # 直前の隠れ状態（stateful運用用・推論時に活用可）
        self.register_buffer("h", torch.zeros(1, n_state, device=dev))  # (batch=1に合わせ後で拡張)

    @torch.no_grad()
    def reset_state(self, batch_size=1):
        self.h = torch.zeros(batch_size, self.n_state, device=self.h.device)

    def forward(self, x, h0=None):
        """
        x: (B, T, d_in)
        return: states per time step (B, T, n_state)
        """
        B, T, _ = x.shape
        device = x.device

        if h0 is None or h0.shape[0] != B:
            h = torch.zeros(B, self.n_state, device=device)
        else:
            h = h0

        a = self.a.view(1, -1)  # (1, n_state)
        Win = self.Win           # (n_state, d_in)
        Wrec = self.Wrec         # (n_state, n_state)

        outputs = []
        act = torch.tanh if self.use_tanh else torch.relu

        for t in range(T):
            xt = x[:, t, :]                    # (B, d_in)
            pre = torch.matmul(h, Wrec.t()) + torch.matmul(xt, Win.t())  # (B, n_state)
            ht = (1 - a) * h + a * act(pre)
            outputs.append(ht.unsqueeze(1))
            h = ht

        return torch.cat(outputs, dim=1)  # (B, T, n_state)


# --------- マルチヘッドESNをconcatしてd_modelへ投影 ----------
class MultiReservoirConcat(nn.Module):
    def __init__(self, d_model=768, n_state_per_head=256, num_heads=8,
                 rho_rec=0.99, sigma_in=1.0, d_connect=32,
                 alpha_min=0.0, alpha_max=1.0, dropout=0.1, device=None,
                 **head_kwargs):
        super().__init__()
        self.heads = nn.ModuleList([
            LeakyESNHead(d_in=d_model, n_state=n_state_per_head,
                         rho_rec=rho_rec, sigma_in=sigma_in, d_connect=d_connect,
                         alpha_min=alpha_min, alpha_max=alpha_max, device=device,
                         **head_kwargs)
            for _ in range(num_heads)
        ])
        total_state = n_state_per_head * num_heads
        self.proj = nn.Linear(total_state, d_model)   # ← ここは学習可（TransformerのW_O相当）
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, d_model)
        states = [head(x) for head in self.heads]              # list of (B, T, n_state)
        h_cat = torch.cat(states, dim=-1)                      # (B, T, H*n_state)
        y = self.proj(h_cat)                                   # (B, T, d_model)
        return self.dropout(y)
# --------- GPT-2 風のブロック（Pre-LN） ----------
class ReservoirBlock(nn.Module):
    def __init__(self, d_model=768, n_state_per_head=256, num_heads=8,
                 mlp_ratio=4, dropout=0.1, mlp_trainable=True, **esn_kwargs):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.reservoir = MultiReservoirConcat(d_model=d_model,
                                              n_state_per_head=n_state_per_head,
                                              num_heads=num_heads,
                                              dropout=dropout, **esn_kwargs)
        self.ln2 = nn.LayerNorm(d_model)
        self.use_mlp = mlp_trainable
        if mlp_trainable:
            hidden = d_model * mlp_ratio
            self.mlp = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, d_model),
                nn.Dropout(dropout),
            )

    def forward(self, x):
        y = x + self.reservoir(self.ln1(x))
        if self.use_mlp:
            y = y + self.mlp(self.ln2(y))
        return y

class GPT2_ESN_LM(nn.Module):
    def __init__(self, vocab_size=50257, n_layer=12, d_model=768, n_state_per_head=256,
                 num_heads=8, max_len=1024, dropout=0.1, mlp_trainable=True,
                 low_rank_out=None,
                 ignore_index=-100,           # ★ 追加: 無視トークン（標準は -100）
                 label_smoothing=0.0,         # ★ 追加: ラベルスムージング（任意）
                 **esn_kwargs):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            ReservoirBlock(d_model=d_model, n_state_per_head=n_state_per_head,
                           num_heads=num_heads, dropout=dropout,
                           mlp_trainable=mlp_trainable, **esn_kwargs)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)

        if low_rank_out is None:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            self.lm_head.weight = self.token_emb.weight
        else:
            r = low_rank_out
            self.A = nn.Linear(r, vocab_size, bias=True)
            self.B = nn.Linear(d_model, r, bias=False)

        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

        self.register_buffer("pos_ids", torch.arange(0, max_len).long(), persistent=False)

    def _to_logits(self, x):
        if hasattr(self, "lm_head"):
            return self.lm_head(x)                 # (B,T,V)
        else:
            return self.A(self.B(x))               # (B,T,V)

    def forward(self, idx, targets=None):
        """
        idx:     (B, T) int64
        targets: (B, T) int64 or None
        return: (logits, loss)  ; loss is None if targets is None
        """
        B, T = idx.shape
        pos = self.pos_ids[:T]
        x = self.token_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = self._to_logits(x)  # (B, T, V)

        loss = None
        if targets is not None:
            # 次トークン予測用にシフト
            # logits: predict t+1 → 対応ラベルは targets[:, 1:]
            # 先頭を捨て、末尾を捨てて整合を取る
            logits_shifted = logits[:, :-1, :].contiguous()    # (B, T-1, V)
            targets_shifted = targets[:, 1:].contiguous()      # (B, T-1)

            # (B*(T-1), V) / (B*(T-1))
            loss = F.cross_entropy(
                logits_shifted.view(-1, logits_shifted.size(-1)),
                targets_shifted.view(-1),
                ignore_index=self.ignore_index,
                label_smoothing=self.label_smoothing,
                reduction="mean",
            )

        return logits, loss


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


@torch.no_grad()
def compute_mean_so_far_ppl(model, blocks, device, ks):
    """
    blocks: list of {"input_ids": LongTensor(T), "labels": LongTensor(T)}  ※B=1相当でOK
    ks:     iterable of int（例: [1,2,4,...,Tmax]）
    return: {k: {"mean_ppl": float}}  各ブロックの平均を返す
    """
    model.eval()
    out = {k: [] for k in ks}
    for blk in blocks:
        ids = blk["input_ids"].to(device).unsqueeze(0)  # (1, T)
        logits, _ = model(ids, targets=None)            # (1, T, V)
        # 次トークン予測にシフト
        lp = F.log_softmax(logits[:, :-1, :], dim=-1)   # (1, T-1, V)
        gold = ids[:, 1:]                                # (1, T-1)
        true_lp = torch.gather(lp, 2, gold.unsqueeze(-1)).squeeze(-1)  # (1, T-1)
        true_lp = true_lp.squeeze(0)                     # (T-1,)
        # 累積平均 -log p → mean_so_far NLL
        cum_lp = torch.cumsum(true_lp, dim=0)            # (T-1,)
        denom = torch.arange(1, cum_lp.numel() + 1, device=ids.device, dtype=cum_lp.dtype)
        mean_nll = -cum_lp / denom                       # (T-1,)
        # 各kに対してPPL = exp(mean_nll[k-1])
        for k in ks:
            if k <= 0:
                continue
            idx = min(k, mean_nll.numel()) - 1
            out[k].append(mean_nll[idx].exp().item())
    # 各kで平均
    return {k: {"mean_ppl": (sum(v) / max(1, len(v)))} for k, v in out.items()}

# =============================================================================
# main
# =============================================================================
# -*- coding: utf-8 -*-
import os, math, time, json, random, argparse, datetime
from pathlib import Path
from itertools import islice

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset

# --- あなたのユーティリティ/データセット/サンプリング関数をここでインポートまたは同ファイルに定義 ---
# BinShardsDataset, StreamingDataset, get_train_loader, get_validation_blocks, safe_decode, sample_sequence
# GPT2_ESN_LM（前メッセージの実装：forwardが (logits, loss) を返す版）

try:
    import wandb
    WANDB_AVAIL = True
except Exception:
    WANDB_AVAIL = False

# =========================================================
# 便利ユーティリティ
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def amp_dtype_for_device():
    if torch.cuda.is_available():
        # bfloat16が使えるGPUはbfloat16、無ければfp16
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return None

# =========================================================
# 引数
# =========================================================
def parse_args():
    p = argparse.ArgumentParser()
    # 基本
    p.add_argument("--tokenizer_path", type=str, default="gpt2", help="HF tokenizer id/path")
    p.add_argument("--dataset_path", type=str, default="HuggingFaceFW/fineweb-edu", help="train dataset (HF path or local dir for shards)")
    p.add_argument("--val_dataset_path", type=str, default="vesteinn/babylm", help="validation dataset (HF path)")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--local_batch_size", type=int, default=64)
    p.add_argument("--total_tokens", type=float, default=1e8)
    p.add_argument("--epochs", type=int, default=1)  # Streaming時は無視

    # モデル（GPT2_ESN_LM）
    p.add_argument("--vocab_size", type=int, default=None, help="未指定ならtokenizerから自動")
    p.add_argument("--n_layer", type=int, default=12)
    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--n_state_per_head", type=int, default=256)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--mlp_trainable", action="store_true", help="pre-LN後のMLPを有効化（Hybrid-ESN）")
    p.add_argument("--low_rank_out", type=int, default=None)

    # ESNハイパラ
    p.add_argument("--spectral_radius", type=float, default=0.99)
    p.add_argument("--sigma_in", type=float, default=1.0)
    p.add_argument("--alpha_min", type=float, default=0.0)
    p.add_argument("--alpha_max", type=float, default=1.0)
    p.add_argument("--d_connect", type=int, default=32)
    p.add_argument("--activation", choices=["tanh", "relu"], default="tanh")

    # 学習
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--grad_clip_norm", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--validate_every_steps", type=int, default=500)
    p.add_argument("--save_every_steps", type=int, default=2000)
    p.add_argument("--generate_every", type=int, default=1000)

    # 生成
    p.add_argument("--gen_temperature", type=float, default=0.8)
    p.add_argument("--gen_top_k", type=int, default=40)
    p.add_argument("--gen_top_p", type=float, default=0.9)
    p.add_argument("--gen_min_p", type=float, default=0.0)
    p.add_argument("--gen_tokens", type=int, default=32)

    # 雑多
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--save_dir", type=str, default="./checkpoints_esn_lm")
    
    # --- mean-so-far PPL (post-train) ---
    p.add_argument("--msf_after_train", action="store_true",
                   help="学習完了後に mean-so-far PPL をまとめて計測する")
    p.add_argument("--msf_seq_len_factor", type=int, default=4,
                   help="MSF評価の最大長 = seq_len * この係数（例: 4 → 4倍長）")
    p.add_argument("--msf_blocks", type=int, default=100,
                   help="MSF評価に使うブロック数の上限（テストセットから切り出し）")
    p.add_argument("--msf_power2", action="store_true",
                   help="k = 1,2,4,...,<=T の冪系列で評価（最後にTを保証）")
    p.add_argument("--msf_points", type=int, default=32,
                   help="power2でない場合、1..T を等間隔に何点で評価するか")
    return p.parse_args()

# =========================================================
# 学習メイン
# =========================================================
def train():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = amp_dtype_for_device()
    print(f"[env] device={device}, amp_dtype={amp_dtype}")

    # tokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    if tok.bos_token is None and tok.eos_token is not None:
        tok.bos_token = tok.eos_token

    vocab_size = args.vocab_size or tok.vocab_size

    # model
    act_tanh = (args.activation == "tanh")
    model = GPT2_ESN_LM(
        vocab_size=vocab_size,
        n_layer=args.n_layer,
        d_model=args.d_model,
        n_state_per_head=args.n_state_per_head,
        num_heads=args.num_heads,
        max_len=args.max_len,
        dropout=args.dropout,
        mlp_trainable=args.mlp_trainable,
        low_rank_out=args.low_rank_out,
        rho_rec=args.spectral_radius,
        sigma_in=args.sigma_in,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        d_connect=args.d_connect,
        use_tanh=act_tanh,
    ).to(device)

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # まずは安定性優先、うまくいかない場合はOFFに

    # dataloaders
    train_loader = get_train_loader(args.tokenizer_path, args.dataset_path, args.seq_len, args.local_batch_size)
    print("[data] train loader ready")

    print("[data] prepare validation blocks…")
    val_ds = load_dataset(args.val_dataset_path, split="test", streaming=False)
    val_blocks = get_validation_blocks(list(islice(val_ds, 100)), tok, args.seq_len, max_blocks=128)
    val_loader = DataLoader(val_blocks, batch_size=args.local_batch_size, num_workers=0, pin_memory=True)
    print(f"[data] val blocks: {len(val_blocks)}")

    # steps & optimizer
    total_steps = math.ceil(args.total_tokens / (args.local_batch_size * args.seq_len))
    warmup = max(1, int(total_steps * args.warmup_ratio))
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup, num_training_steps=total_steps)

    # wandb
    use_wandb = WANDB_AVAIL and (args.wandb_project is not None)
    if use_wandb:
        run_name = args.run_name or f"ESN-GPT2-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    # loop
    os.makedirs(args.save_dir, exist_ok=True)
    model.train()
    start = time.time()
    seen_tokens = 0
    max_mem_mb = 0.0

    scaler = None  # bfloat16 では通常不要。fp16でoverflowが出る場合のみGradScalerを導入。

    step = 0
    for batch in train_loader:
        step += 1
        ids = batch.to(device)

        with torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype else torch.no_grad():
            # autocast が無いCPU時は no_grad にしない（学習が止まる）ので分岐に注意
            if amp_dtype is None:
                logits, loss = model(ids, targets=ids)
            else:
                logits, loss = model(ids, targets=ids)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if args.grad_clip_norm and args.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

        optimizer.step()
        scheduler.step()

        # stats
        seen_tokens += ids.numel()
        vram_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
        max_mem_mb = max(max_mem_mb, vram_mb)

        if use_wandb and (step % 10 == 0):
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/ppl": math.exp(min(20.0, loss.item())),  # 発散防止
                    "lr": scheduler.get_last_lr()[0],
                    "tokens/seen": seen_tokens,
                    "gpu/max_mem_mb": max_mem_mb,
                    "step": step,
                    "total_steps": total_steps,
                },
                step=step,
            )

        # validate
        if step % args.validate_every_steps == 0:
            model.eval()
            v_losses = []
            with torch.no_grad():
                for v in islice(val_loader, len(val_loader)):
                    v_ids = v["input_ids"].to(device)
                    _, v_loss = model(v_ids, targets=v["labels"].to(device))
                    v_losses.append(v_loss.item())
            val_loss = sum(v_losses) / max(1, len(v_losses))
            if use_wandb:
                wandb.log({"val/loss": val_loss, "val/ppl": math.exp(min(20.0, val_loss))}, step=step)
            print(f"[val] step={step}  loss={val_loss:.4f}  ppl={math.exp(val_loss):.2f}")
            model.train()

        # save
        if step % args.save_every_steps == 0:
            ckpt = os.path.join(args.save_dir, f"ckpt_step{step}.pt")
            torch.save({"model": model.state_dict(), "step": step, "args": vars(args)}, ckpt)
            print(f"[ckpt] saved: {ckpt}")

        # generate
        if step % args.generate_every == 0:
            model.eval()
            for prompt in ["Hello,", "I am", "科学とは"]:
                ids0 = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
                gen = sample_sequence(
                    model, ids0, max_new_tokens=args.gen_tokens,
                    temperature=args.gen_temperature, top_k=args.gen_top_k,
                    top_p=args.gen_top_p, min_p=args.gen_min_p, device=device
                )
                text = safe_decode(gen[0].tolist(), tok)
                print(f"[gen] {prompt} >>> {text[:200]}")
                if use_wandb:
                    wandb.log({f"gen/{prompt}": wandb.Html(f"<b>{prompt}</b> {text}")}, step=step)
            model.train()

        # end condition
        if seen_tokens >= args.total_tokens or step >= total_steps:
            print("[train] reached token/step budget. stopping.")
            break

    # final save
    final = os.path.join(args.save_dir, "final.pt")
    torch.save({"model": model.state_dict(), "step": step, "args": vars(args)}, final)
    print(f"[done] saved final: {final}  steps={step}  time={(time.time()-start):.1f}s")

    # ===============================
    # Post-Train: mean-so-far PPL
    # ===============================
    if args.msf_after_train:
        print("▶️ Post-train evaluation: mean-so-far PPL ...")
        device = next(model.parameters()).device
        # 評価する最大系列長（seq_len の倍長）
        seq_len_test = max(2, args.seq_len * max(1, args.msf_seq_len_factor))

        # テスト/検証データからブロック作成
        tok = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token or tok.unk_token

        test_ds = load_dataset(args.val_dataset_path, split="test", streaming=False)
        buffer = []
        test_blocks = []
        for ex in test_ds:
            toks = tok(ex["text"], return_attention_mask=False)["input_ids"]
            buffer.extend(toks)
            while len(buffer) >= seq_len_test and len(test_blocks) < args.msf_blocks:
                blk = buffer[:seq_len_test]
                test_blocks.append({
                    "input_ids": torch.tensor(blk, dtype=torch.long),
                    "labels":   torch.tensor(blk, dtype=torch.long),
                })
                buffer = buffer[seq_len_test:]
            if len(test_blocks) >= args.msf_blocks:
                break

        if not test_blocks:
            print("⚠️ MSF-PPL用のブロックが作成できませんでした（データ不足）。スキップします。")
        else:
            # k の取り方
            Tm = seq_len_test
            if args.msf_power2:
                ks = []
                k = 1
                while k <= Tm:
                    ks.append(k)
                    k *= 2
                if ks[-1] != Tm:
                    ks.append(Tm)
            else:
                grid = torch.linspace(1, Tm, steps=max(2, args.msf_points)).round().long().tolist()
                ks = sorted(set(int(k) for k in grid))

            # 推論メモリ・速度の計測（任意）
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            msf = compute_mean_so_far_ppl(model, test_blocks, device, ks)
            t1 = time.time()
            inf_time = t1 - t0
            total_tokens = len(test_blocks) * Tm
            tok_per_sec = total_tokens / inf_time if inf_time > 0 else 0.0
            peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0.0

            # 保存＆ログ
            os.makedirs("./reports", exist_ok=True)
            report = {
                "msf_seq_len": Tm,
                "msf_num_blocks": len(test_blocks),
                "msf_time_sec": inf_time,
                "msf_tokens_per_sec": tok_per_sec,
                "msf_peak_mem_mb": peak_mb,
                "msf_curve": {int(k): float(msf[k]["mean_ppl"]) for k in ks},
            }
            rep_path = os.path.join("./reports", f"msf_report_{int(time.time())}.json")
            with open(rep_path, "w") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"📄 MSF-PPL report saved: {rep_path}")

            # W&B（任意）
            if WANDB_AVAIL and args.wandb_project is not None:
                table = wandb.Table(columns=["k", "mean_so_far_ppl"])
                for k in ks:
                    table.add_data(int(k), float(msf[k]["mean_ppl"]))
                wandb.log({
                    "post_train/msf_table": table,
                    "post_train/msf_time_sec": inf_time,
                    "post_train/msf_tokens_per_sec": tok_per_sec,
                    "post_train/msf_peak_mem_mb": peak_mb,
                })

    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    train()
