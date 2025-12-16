import torch
import torch.nn as nn
import torch.nn.functional as F

class NGRCBlock(nn.Module):
    """
    1層分の NGRC ブロック。
    入力:  h (B, T, d_model)
    処理:  lag 窓で遅延ベクトル z (B, T, L, d_model) を作り、
           φ(z) = [1, z, z^2, ..., z^P, (optional cross)]
           φ -> (低ランク)線形 -> Δh
           ゲートの種類に応じて h と組み合わせる
    出力:  h_next (B, T, d_model)
    """

    def __init__(
        self,
        d_model: int,
        lag: int,
        poly_degree: int = 2,
        max_cross_terms: int = 0,
        readout_rank: int | None = 512,
        dropout: float = 0.1,
        gating: str | None = None,   # None/"none", "layer", "gru", "lag_degree"
    ):
        super().__init__()
        assert poly_degree >= 1
        assert lag >= 1
        self.d_model = d_model
        self.lag = lag
        self.poly_degree = poly_degree
        self.max_cross_terms = max_cross_terms
        self.readout_rank = readout_rank
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)
        self.gating = (gating or "none").lower()

        base_dim = d_model * lag
        self.base_dim = base_dim

        # cross term 用 index（flattened z: 長さ base_dim 上の index）
        if max_cross_terms > 0:
            self.register_buffer("cross_i", torch.randint(0, base_dim, (max_cross_terms,)))
            self.register_buffer("cross_j", torch.randint(0, base_dim, (max_cross_terms,)))
        else:
            self.cross_i = None
            self.cross_j = None

        # φ の次元: 1 (bias) + base_dim * poly_degree + cross_dim
        phi_dim = 1 + base_dim * poly_degree + max_cross_terms
        self.phi_dim = phi_dim

        # φ → d_model の線形（低ランク or 1段）
        if readout_rank is None:
            self.proj = nn.Linear(phi_dim, d_model, bias=False)
            self.proj1 = None
            self.proj2 = None
        else:
            self.proj1 = nn.Linear(phi_dim, readout_rank, bias=False)
            self.proj2 = nn.Linear(readout_rank, d_model, bias=False)
            self.proj = None

        # gating 用のサブネット
        if self.gating == "layer":
            # 層出力ゲート: g(h) ∈ (0,1)^{d_model}
            self.gate_layer = nn.Linear(d_model, d_model, bias=True)

        elif self.gating == "gru":
            # GRU 風 update ゲート: z_gate(h) ∈ (0,1)^{d_model}
            self.gate_z = nn.Linear(d_model, d_model, bias=True)

        elif self.gating == "lag_degree":
            # 各 token ごとに L + P 個のゲート
            #   - 前半 L 個: lag ごとのゲート
            #   - 後半 P 個: degree ごとのゲート
            self.lagdeg_gate = nn.Linear(d_model, lag + poly_degree, bias=True)

        elif self.gating in ("none",):
            pass
        else:
            raise ValueError(f"unknown gating mode: {gating}")

    # (B,T,D) → (B,T,L,D)
    def _build_lagged(self, h: torch.Tensor) -> torch.Tensor:
        B, T, D = h.shape
        L = self.lag
        if L == 1:
            z = h.unsqueeze(2)  # (B,T,1,D)
        else:
            pad = h[:, :1, :].expand(B, L - 1, D)      # (B,L-1,D)
            h_padded = torch.cat([pad, h], dim=1)      # (B,T+L-1,D)
            # unfold: (B, T, D, L) になるので permute で (B,T,L,D) に
            z = h_padded.unfold(dimension=1, size=L, step=1)  # (B,T,D,L)
            z = z.permute(0, 1, 3, 2).contiguous()            # (B,T,L,D)
        return z

    # z: (B,T,L,D) → φ: (B,T,phi_dim)
    def _build_phi(self, z: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:
        B, T, L, D = z.shape
        feats = []

        # 1) bias
        ones = torch.ones(B, T, 1, dtype=z.dtype, device=z.device)
        feats.append(ones)

        # 2) lag/degree gating の準備（必要な場合）
        if self.gating == "lag_degree":
            assert h is not None
            # (B,T,L+P)
            gates = torch.sigmoid(self.lagdeg_gate(h))
            gate_lag = gates[..., :L]                   # (B,T,L)
            gate_deg = gates[..., L:]                   # (B,T,P)
        else:
            gate_lag = gate_deg = None

        # 3) z^1, z^2, ..., z^P を追加
        z_power = z  # degree=1
        for deg in range(1, self.poly_degree + 1):
            if deg > 1:
                z_power = z_power * z  # 要素ごとの乗算で z^deg

            if self.gating == "lag_degree":
                # lag ごと + degree ごとのゲートをかける
                g_l = gate_lag                      # (B,T,L)
                g_d = gate_deg[..., deg - 1]        # (B,T)
                g = g_l * g_d.unsqueeze(-1)         # (B,T,L)
                g = g.unsqueeze(-1)                 # (B,T,L,1)
                z_eff = z_power * g                 # (B,T,L,D)
            else:
                z_eff = z_power                     # (B,T,L,D)

            feats.append(z_eff.reshape(B, T, L * D))

        # 4) cross term（使うなら degree=1 の z から）
        if self.max_cross_terms > 0 and self.cross_i is not None:
            z1_flat = z.reshape(B, T, L * D)        # degree=1 の z
            z_i = z1_flat[..., self.cross_i]        # (B,T,M)
            z_j = z1_flat[..., self.cross_j]        # (B,T,M)
            feats.append(z_i * z_j)                 # (B,T,M)

        phi = torch.cat(feats, dim=-1)
        assert phi.shape[-1] == self.phi_dim
        return phi

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B,T,d_model)
        戻り: (B,T,d_model)
        """
        z = self._build_lagged(h)  # (B,T,L,D)
        if self.gating == "lag_degree":
            phi = self._build_phi(z, h=h)
        else:
            phi = self._build_phi(z, h=None)

        # φ → d_model
        if self.proj is not None:
            delta = self.proj(phi)
        else:
            mid = self.proj1(phi)
            delta = self.proj2(mid)
        delta = self.dropout(delta)

        # gating の種類ごとに残差の作り方を変える
        if self.gating in ("none",):
            # ただの残差
            h_next = h + delta

        elif self.gating == "layer":
            # 層出力ゲート: g(h) ⊙ Δh
            gate = torch.sigmoid(self.gate_layer(h))  # (B,T,D)
            h_next = h + gate * delta

        elif self.gating == "gru":
            # GRU 風 update ゲート: h_new = (1-z)⊙h + z⊙tanh(Δh)
            z_gate = torch.sigmoid(self.gate_z(h))    # (B,T,D)
            candidate = torch.tanh(delta)             # (B,T,D)
            h_next = (1.0 - z_gate) * h + z_gate * candidate

        elif self.gating == "lag_degree":
            # lag/degree ゲートは φ 側に掛けているので、ここは普通の残差
            h_next = h + delta

        else:
            raise RuntimeError("unreachable gating mode")

        return self.ln(h_next)

class NGRC_LM(nn.Module):
    """
    多層 NGRC ベースの言語モデル。

    - Embedding: token → h_0 ∈ R^{d_model}
    - NGRCBlock を n_layers 個 stacked
    - 最後に LayerNorm & LM head (d_model → vocab)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        lag: int = 4,
        poly_degree: int = 2,
        max_cross_terms: int = 0,
        readout_rank: int | None = 512,
        gating: str | None = None,      # None/"none", "layer", "gru", "lag_degree"
        embed_trainable: bool = True,
        loss_type: str = "ce",          # "ce" or "mse"
    ):
        super().__init__()
        self.arch_name = "multilayer_NGRC"
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.loss_type = loss_type

        # 埋め込み
        self.embed = nn.Embedding(vocab_size, d_model)
        if not embed_trainable:
            for p in self.embed.parameters():
                p.requires_grad = False

        # 多層 NGRC ブロック
        self.blocks = nn.ModuleList([
            NGRCBlock(
                d_model=d_model,
                lag=lag,
                poly_degree=poly_degree,
                max_cross_terms=max_cross_terms,
                readout_rank=readout_rank,
                dropout=0.1,
                gating=gating,
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.LongTensor, labels: torch.LongTensor | None = None):
        """
        x: (B, T) token ids
        labels: (B, T) or None
        """
        device = next(self.parameters()).device
        x = x.to(device)
        h = self.embed(x)  # (B,T,d_model)

        for blk in self.blocks:
            h = blk(h)

        h = self.ln_f(h)
        logits = self.lm_head(h)  # (B,T,vocab)

        if labels is None:
            return logits

        labels = labels.to(device)
        B, T = x.shape
        V = self.vocab_size

        logits_shift = logits[:, :-1, :].reshape(-1, V)
        targets = labels[:, 1:].reshape(-1)

        if self.loss_type == "ce":
            loss = F.cross_entropy(logits_shift, targets)
        elif self.loss_type == "mse":
            one_hot = F.one_hot(targets, num_classes=V).to(logits_shift.dtype)
            probs = F.softmax(logits_shift, dim=-1)
            loss = F.mse_loss(probs, one_hot)
        else:
            raise ValueError(f"unknown loss_type: {self.loss_type}")

        return logits, loss




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
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
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
    """Load pre-tokenised uint16 shards with zero-copy mmap (rank & worker sharded)."""

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

    def __iter__(self):
        worker = get_worker_info()
        worker_id, num_workers = (worker.id, worker.num_workers) if worker else (0, 1)

        for i, f in enumerate(self.files):
            global_idx = self.rank * num_workers + worker_id
            global_stride = self.world_size * num_workers
            if i % global_stride != global_idx:
                continue
            yield from self._yield_file_tokens(f)


class StreamingDataset(IterableDataset):
    def __init__(self, dataset_path, split, tokenizer, seq_len):
        self.ds = load_dataset(dataset_path, split=split, streaming=True).shuffle(
            buffer_size=10, seed=42
        )
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


def prepare_hf_repo_id(write_token, explicit_repo, default_prefix="NGRC-LM"):
    api = HfApi(token=write_token)
    if explicit_repo:
        return explicit_repo, api
    owner = api.whoami()["name"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    repo_id = f"{owner}/{default_prefix}-{timestamp}"
    return repo_id, api


def _named_params(model):
    try:
        return model.module.named_parameters()
    except AttributeError:
        return model.named_parameters()


@torch.no_grad()
def log_gradients_wandb(model, step: int, tag: str = "preclip", topk: int = 5):
    try:
        import wandb
    except Exception:
        return
    if wandb.run is None:
        return

    total_sq = 0.0
    total_elems = 0
    nan_count = 0
    inf_count = 0
    zero_count = 0
    layer_norms = []

    for name, p in _named_params(model):
        g = p.grad
        if g is None:
            continue
        gf = g.detach()
        gcpu = gf.to(dtype=torch.float32, device="cpu")
        nans = torch.isnan(gcpu).sum().item()
        infs = torch.isinf(gcpu).sum().item()
        zeros = (gcpu == 0).sum().item()
        nan_count += nans
        inf_count += infs
        zero_count += zeros
        total_elems += gcpu.numel()

        ln = torch.linalg.norm(gcpu).item()
        layer_norms.append((ln, name))
        total_sq += ln * ln

    global_grad_norm = math.sqrt(max(total_sq, 0.0))
    zero_frac = (zero_count / max(total_elems, 1)) if total_elems > 0 else 0.0

    layer_norms.sort(reverse=True, key=lambda x: x[0])
    top_items = {f"grad_top/{i}_{name}": val for i, (val, name) in enumerate(layer_norms[:topk])}

    payload = {
        f"grad/{tag}/global_norm": global_grad_norm,
        f"grad/{tag}/nan_count": nan_count,
        f"grad/{tag}/inf_count": inf_count,
        f"grad/{tag}/zero_frac": zero_frac,
        f"grad/{tag}/layers": len(layer_norms),
        **top_items,
    }
    wandb.log(payload, step=step)
# =============================================================================
# Sampling util
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
        # logits の異常値をチェックしてクリップ
        logits = torch.clamp(logits, min=-1e4, max=1e4)
        logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
        logits = torch.where(torch.isinf(logits), torch.zeros_like(logits), logits)
        next_logits = logits[:, -1, :] / max(temperature, 1e-8)
        if top_k > 0:
            values, _ = torch.topk(next_logits, top_k)
            min_value = values[:, -1].unsqueeze(-1)
            next_logits = torch.where(
                next_logits < min_value,
                torch.full_like(next_logits, float("-inf")),
                next_logits,
            )
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumulative_probs > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_logits[mask] = float("-inf")
            next_logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)
        # 最終的な logits の異常値をチェック
        next_logits = torch.clamp(next_logits, min=-1e4, max=1e4)
        next_logits = torch.where(torch.isnan(next_logits), torch.zeros_like(next_logits), next_logits)
        next_logits = torch.where(torch.isinf(next_logits), torch.zeros_like(next_logits), next_logits)
        
        if min_p > 0.0:
            probs = F.softmax(next_logits, dim=-1)
            next_logits = torch.where(
                probs < min_p, torch.full_like(next_logits, float("-inf")), next_logits
            )
        probs = F.softmax(next_logits, dim=-1)
        # 確率の異常値をチェック（負の値や inf/nan を防ぐ）
        probs = torch.clamp(probs, min=0.0, max=1.0)
        probs = torch.where(torch.isnan(probs), torch.ones_like(probs) / probs.size(-1), probs)
        probs = torch.where(torch.isinf(probs), torch.ones_like(probs) / probs.size(-1), probs)
        # 正規化して確率の合計が1になるようにする
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)
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
        ds = StreamingDataset(path, "train", tok, seq_len)
        print("[data] HF streaming dataset", path)

    num_workers = 2 if Path(path).is_dir() else 6
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


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
                "labels": torch.tensor(block, dtype=torch.long),
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
# Argument parser (NGRC version)
# =============================================================================
def parse_args_ngrc():
    parser = argparse.ArgumentParser()

    # dist / DS
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    parser.add_argument("--deepspeed_config", type=str, default="ds_config.json")
    parser.add_argument("--use_deepspeed", action="store_true")

    # training hyperparameters
    parser.add_argument("--local_batch_size", type=int, default=50)
    parser.add_argument("--use_gpu_amount", type=int, default=max(torch.cuda.device_count(), 1))
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--validate_every_steps", type=int, default=200)
    parser.add_argument("--save_checkpoint_every_steps", type=int, default=200)
    parser.add_argument("--generate_every", type=int, default=1000)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--total_tokens", type=float, default=100e6)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--grad_clip_norm", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--tokenizer_path", type=str, default="meta-llama/Llama-2-7b-hf")

    # NGRC hyperparams
    parser.add_argument("--ngrc_d_model", type=int, default=64 ,help="埋め込み次元数")
    parser.add_argument("--ngrc_lag", type=int, default=32, help="ラグ")
    parser.add_argument("--ngrc_poly_degree", type=int, default=2, help="多項式次数")
    parser.add_argument("--ngrc_max_cross_terms", type=int, default=256, help="クロス項の最大数")
    parser.add_argument("--ngrc_readout_rank", type=int, default=512, help="低ランク readout 次元")
    parser.add_argument("--ngrc_num_layers", type=int, default=6, help="NGRC ブロックの層数")
    parser.add_argument(
        "--ngrc_gating",
        type=str,
        default="none",
        choices=["none", "layer", "gru", "lag_degree"],
        help="NGRC ブロックごとのゲート方式を選択",
    )
    parser.add_argument(
        "--ngrc_embed_frozen",
        action="store_true",
        help="Freeze embedding weights (default: trainable).",
    )
    parser.add_argument(
        "--ngrc_training",
        choices=["sgd", "ridge"],
        default="sgd",
        help="SGD: usual LM training, ridge: closed-form NGRC readout fit.",
    )
    parser.add_argument(
        "--ngrc_loss",
        choices=["ce", "mse"],
        default="ce",
        help="Loss type for SGD training (cross-entropy or MSE-on-probs).",
    )
    parser.add_argument(
        "--ngrc_ridge_alpha",
        type=float,
        default=1e-3,help="Ridge 正則化係数 (ngrc_training='ridge' の場合)",

    )
    parser.add_argument(
        "--ngrc_ridge_max_batches",
        type=int,
        default=200,
        help="Max train batches used for ridge fitting.",
    )

    # Dataset
    parser.add_argument("--dataset_path", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--val_dataset_path", type=str, default="vesteinn/babylm")

    # Weights & Biases
    parser.add_argument("--wandb_project", type=str, default="NGRC_LanguageModel")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--api_file", type=str, default="api.txt")

    parser.add_argument("--report_path", type=str, default="../reports_multilayer_NGRC")
    # HF
    parser.add_argument("--hf_repo", type=str, default=None)
    parser.add_argument("--hf_private", action="store_false")

    # JIT compile (一応残しておくが、sparse は使っていないので安定)
    parser.add_argument("--enable_compile", action="store_true")


    return parser.parse_args()


# =============================================================================
# main experiment
# =============================================================================
def NGRC_experiment(lr):
    @torch.no_grad()
    def compute_mean_so_far_ppl(model, blocks, device, ks):
        results = {k: [] for k in ks}
        for block in tqdm(blocks, desc="Computing mean-so-far PPL"):
            ids = block["input_ids"].to(device).unsqueeze(0)
            logits = model(ids)
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            true_lp = torch.gather(log_probs, 2, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            cum_lp = torch.cumsum(true_lp.squeeze(0), dim=0)
            lengths = torch.arange(1, cum_lp.size(0) + 1, device=device)
            mean_nll = -cum_lp / lengths
            for k in ks:
                idx = min(k, mean_nll.size(0)) - 1
                results[k].append(mean_nll[idx].exp().item())
        return {k: {"mean_ppl": sum(v) / len(v)} for k, v in results.items()}

    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args = parse_args_ngrc()
    print(args)

    seed = random.randint(1000000, 5000000)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    WANDB_AVAILABLE = False
    api_keys = load_api_keys(args.api_file)
    if "WANDB_API_KEY" in api_keys:
        os.environ["WANDB_API_KEY"] = api_keys["WANDB_API_KEY"]
    if "HF_READ_TOKEN" in api_keys:
        from huggingface_hub import login
        login(token=api_keys["HF_WRITE_TOKEN"])

    # ridge のときは単純に 1 プロセス想定
    if args.ngrc_training == "ridge":
        distributed = False
    else:
        if args.use_deepspeed and _HAS_DS and torch.cuda.device_count() > 1:
            torch.cuda.set_device(args.local_rank)
            distributed = True
        else:
            distributed = False

    world_size = dist.get_world_size() if (distributed and dist.is_initialized()) else 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.bos_token = tokenizer.eos_token

    embed_trainable = not args.ngrc_embed_frozen

    model = NGRC_LM(
        vocab_size=tokenizer.vocab_size,
        d_model=args.ngrc_d_model,
        n_layers=args.ngrc_num_layers,       # 新しい引数
        lag=args.ngrc_lag,
        poly_degree=args.ngrc_poly_degree,   # 新しい引数 (default=2)
        max_cross_terms=args.ngrc_max_cross_terms,
        readout_rank=512,                    # 固定でも引数でもOK
        gating=args.ngrc_gating,             # "none" / "layer" / "gru" / "lag_degree"
        embed_trainable=not args.ngrc_embed_frozen,
        loss_type=args.ngrc_loss,
    ).to(device).to(torch.bfloat16)
    if args.enable_compile:
        model = torch.compile(model)

    param_millions = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"parameter count: {param_millions:.2f}M")

    optimizer = None
    scheduler = None

    if args.ngrc_training == "sgd":
        if distributed:
            model, optimizer, _, _ = deepspeed.initialize(
                args=args,
                model=model,
                model_parameters=model.parameters(),
            )
            scheduler = None
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                betas=(args.beta1, args.beta2),
                weight_decay=args.weight_decay,
            )
            total_steps = math.ceil(
                args.total_tokens / (args.local_batch_size * args.use_gpu_amount * args.seq_len)
            )
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=math.ceil(0.1 * total_steps),
                num_training_steps=total_steps,
            )
            model.to(device)

    train_loader = get_train_loader(
        args.tokenizer_path, args.dataset_path, args.seq_len, args.local_batch_size
    )

    val_ds = load_dataset(args.val_dataset_path, split="test", streaming=False)
    val_blocks = get_validation_blocks(list(islice(val_ds, 100)), tokenizer, args.seq_len, 1)
    val_loader = DataLoader(
        val_blocks, batch_size=args.local_batch_size, num_workers=0, pin_memory=True
    )

    max_steps = math.ceil(
        args.total_tokens / (args.local_batch_size * args.use_gpu_amount * args.seq_len)
    )

    tokens_seen_local = 0
    tokens_seen_global = 0
    success_flag = True
    val_loss = 10.75

    start_time = time.time()
    max_mem_mb = 0.0
    generated_text_log = []
    model.train()

    # WandB init
    if (not distributed) or args.local_rank == 0:
        run_name = args.wandb_run_name or (
            f"{model.arch_name}({param_millions:.2f}M_d{args.ngrc_d_model}"
            f"_lag{args.ngrc_lag}_poly{args.ngrc_poly_degree}_rank{args.ngrc_readout_rank}_lr{args.learning_rate}"
            f"_bs{args.local_batch_size}_seq{args.seq_len}_{run_id}"
        )
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        WANDB_AVAILABLE = True

    # ================= NGRC ridge training mode ========================
    if args.ngrc_training == "ridge":
        if (not distributed) or args.local_rank == 0:
            print("[NGRC] using closed-form ridge to fit readout (no SGD loop).")
        fit_ngrc_ridge_readout(
            model,
            train_loader,
            alpha=args.ngrc_ridge_alpha,
            max_batches=args.ngrc_ridge_max_batches,
            device=device,
        )
        model.eval()
        val_loss_list = []
        with torch.no_grad():
            for v_batch in val_loader:
                v_ids = v_batch["input_ids"].to(device)
                v_labels = v_batch["labels"].to(device)
                _, v_loss = model(v_ids, labels=v_labels)
                val_loss_list.append(v_loss.item())
        val_loss = sum(val_loss_list) / len(val_loss_list)
        if (not distributed) or args.local_rank == 0:
            print(
                f"[NGRC ridge] validation loss after ridge fit: {val_loss:.4f}, "
                f"ppl={math.exp(val_loss):.2f}"
            )
            wandb.log(
                {"val_loss": val_loss, "val_perplexity": math.exp(val_loss)},
                step=0,
            )
        # そのまま「学習後の処理」に進む
    else:
        # ==================== 通常 SGD ループ ==========================
        for step, batch in enumerate(train_loader, start=1):
            ids = batch.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, loss = model(ids, labels=ids)

            if distributed:
                model.backward(loss)
                log_gradients_wandb(model, step, tag="preclip")
                # DeepSpeed の global grad norm をログしたければここに追加

                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.module.parameters(), args.grad_clip_norm)
                    log_gradients_wandb(model, step, tag="postclip")

                model.step()
                current_lr = model.get_lr()[0]
            else:
                loss.backward()
                log_gradients_wandb(model, step, tag="preclip")

                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                    log_gradients_wandb(model, step, tag="postclip")

                optimizer.step()
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                optimizer.zero_grad()

            step_time = time.time() - start_time
            tokens_seen_local += args.local_batch_size * args.seq_len
            tokens_seen_global += args.local_batch_size * args.seq_len * world_size
            tokens_per_sec_global = tokens_seen_global / step_time if step_time > 0 else 0.0
            tokens_per_sec_local = tokens_seen_local / step_time if step_time > 0 else 0.0
            vram_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
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
                        _, v_loss = model(v_ids, labels=v_labels)
                        val_loss_list.append(v_loss.item())
                val_loss = sum(val_loss_list) / len(val_loss_list)
                if (not distributed) or args.local_rank == 0:
                    wandb.log(
                        {
                            "val_loss": val_loss,
                            "val_perplexity": math.exp(val_loss),
                        },
                        step=step,
                    )
                model.train()

            if step % args.save_checkpoint_every_steps == 0 and (
                (not distributed) or args.local_rank == 0
            ):
                ckpt_name = f"checkpoint_step{step}_tokens{tokens_seen_global}.pt"
                save_dir = (
                    f"./checkpoint_ngrc/NGRC_LM({param_millions:.2f}M_d{args.ngrc_d_model}"
                    f"_lag{args.ngrc_lag}_poly{args.ngrc_poly_degree}_rank{args.ngrc_readout_rank}_{run_id}"
                )
                os.makedirs(save_dir, exist_ok=True)
                if distributed:
                    model.save_checkpoint(save_dir=f"./{save_dir}/{ckpt_name}", tag=f"step_{step}")
                else:
                    torch.save(model.state_dict(), f"./{save_dir}/{ckpt_name}")

            if step % args.generate_every == 0 and ((not distributed) or args.local_rank == 0):
                model.eval()
                for prompt in ["Large Language model is "]:
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
                    wandb.log({"generated": wandb.Html(f"<b>{prompt}</b>{output_str}")}, step=step)
                    generated_text_log.append(output_str)
                model.train()

            if step % 500 == 0:
                print(
                    f"[log] train loss: {loss.item()}, step: {step}, "
                    f"tokens_seen_global: {tokens_seen_global}, "
                    f"tokens_per_sec_global: {tokens_per_sec_global}, max_mem_mb: {max_mem_mb}"
                )

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

        if "HF_WRITE_TOKEN" in api_keys and success_flag:
            repo_id, api = prepare_hf_repo_id(
                api_keys["HF_WRITE_TOKEN"],
                args.hf_repo,
                default_prefix=f"NGRC_LM_{param_millions:.2f}M",
            )
            api.create_repo(repo_id=repo_id, exist_ok=True, private=args.hf_private)
            upload_folder(
                repo_id=repo_id,
                folder_path=final_dir,
                path_in_repo=".",
                token=api_keys["HF_WRITE_TOKEN"],
                ignore_patterns=["*.pt"],
            )
            print(f"✅ Model pushed to https://huggingface.co/{repo_id}")
        else:
            print("HF upload skipped (token absent or repo not specified).")

        param_count = sum(p.numel() for p in model.parameters())
        report = {
            "run_name": wandb.run.name if WANDB_AVAILABLE else "offline_run",
            "arch_name": model.arch_name,
            "hyperparameters": vars(args),
            "parameter_count": param_count,
            "max_gpu_memory_MB": max_mem_mb,
            "training_time_sec": total_train_time,
            "final_train_loss": loss.item() if args.ngrc_training == "sgd" else None,
            "final_train_perplexity": math.exp(loss.item())
            if args.ngrc_training == "sgd"
            else None,
            "final_val_loss": val_loss if "val_loss" in locals() else None,
            "final_val_perplexity": math.exp(val_loss)
            if "val_loss" in locals()
            else None,
            "generated": generated_text_log,
            "seed": seed,
        }

    # 追加の inference benchmark & mean-so-far PPL
    if success_flag and is_master:
        test_input = torch.randint(0, tokenizer.vocab_size, (1, args.seq_len), device=device)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t_inf_start = time.time()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _ = model(test_input)
        infer_time = time.time() - t_inf_start
        infer_tok_per_sec = args.seq_len / infer_time if infer_time > 0 else 0.0
        infer_mem_mb = (
            torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
        )
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
                test_blocks.append(
                    {
                        "input_ids": torch.tensor(blk, dtype=torch.long),
                        "labels": torch.tensor(blk, dtype=torch.long),
                    }
                )
                buffer = buffer[seq_len_test:]
        test_blocks = test_blocks[:100]
        ks = list(range(1, seq_len_test + 1))

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t_inf_start = time.time()
        mean_so_far = compute_mean_so_far_ppl(
            model.module if distributed else model,
            test_blocks,
            device,
            ks,
        )
        t_inf_end = time.time()
        inf_time = t_inf_end - t_inf_start
        total_inf_tokens = len(test_blocks) * seq_len_test
        infer_tok_per_sec = total_inf_tokens / inf_time if inf_time > 0 else 0.0
        infer_mem_mb = (
            torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
        )
        report.update({"inference_tok_per_sec": infer_tok_per_sec, "inference_mem_MB": infer_mem_mb})
        print(
            f"Inference time: {inf_time:.2f}s, Tokens/sec: {infer_tok_per_sec:.2f}, "
            f"Memory: {infer_mem_mb:.2f}MB"
        )
        if WANDB_AVAILABLE:
            wandb.log(
                {
                    "inference_time": inf_time,
                    "inference_tok_per_sec": infer_tok_per_sec,
                    "inference_mem_MB": infer_mem_mb,
                }
            )

        report_ks = []
        k = 1
        while k <= seq_len_test:
            report_ks.append(k)
            k *= 2
        report_ks = [k for k in report_ks if k <= seq_len_test]

        if WANDB_AVAILABLE:
            table = wandb.Table(columns=["token_length", "perplexity"])
            for k in report_ks:
                table.add_data(k, mean_so_far[k]["mean_ppl"])
            chart = wandb.plot.line(
                table,
                x="token_length",
                y="perplexity",
                title="Mean-so-far PPL vs Token Length",
            )
            wandb.log({"mean_so_far_ppl_curve": chart})

        report["test_mean_so_far_ppl_curve"] = {k: mean_so_far[k]["mean_ppl"] for k in ks}

        report_path = f"./{args.report_path}/{report['run_name']}_report.txt"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=2))
        print(f"📄 Training report written to {report_path}")

    if WANDB_AVAILABLE:
        wandb.finish()

    del model
    if optimizer is not None:
        del optimizer
    if scheduler is not None:
        del scheduler
    del train_loader
    del val_loader
    if "test_blocks" in locals():
        del test_blocks
    if "test_ds" in locals():
        del test_ds


import gc
def main():
    args = parse_args_ngrc()
    NGRC_experiment(lr=args.learning_rate)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()