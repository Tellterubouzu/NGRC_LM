#!/usr/bin/env python3
"""
train template ‑‑ **fixed version**
  - ESN_mlr 改良実装（スペクトル半径・one‑hot gather など修正）
  - IterableDataset のワーカ重複防止
  - 分散環境での token 合算を all_reduce で厳密化
  - compute_mean_so_far_ppl の返値を {k:{"mean_ppl":…}} 形式に変更
  - その他細部を調整
"""
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


def prepare_hf_repo_id(write_token, explicit_repo, a_min, a_max, default_prefix="ESN-lr-experimtn"):
    api = HfApi(token=write_token)
    if explicit_repo:
        return explicit_repo, api
    owner = api.whoami()["name"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    repo_id = f"{owner}/{default_prefix}-a_min{a_min}-a_max{a_max}-{timestamp}"
    return repo_id, api

# ---- Grad logging utils ------------------------------------------------------
def _named_params(model):
    try:
        # DeepSpeed Engine のときは中身にアクセス
        return model.module.named_parameters()
    except AttributeError:
        return model.named_parameters()

@torch.no_grad()
def log_gradients_wandb(model, step: int, tag: str = "preclip", topk: int = 5):
    """全勾配のL2ノルムや異常値、主要レイヤの勾配ノルムを wandb.log する"""
    try:
        import wandb
    except Exception:
        return  # wandb未使用なら何もしない
    if wandb.run is None:
        return

    total_sq = 0.0
    total_elems = 0
    nan_count = 0
    inf_count = 0
    zero_count = 0

    layer_norms = []  # (||g||2, name)

    for name, p in _named_params(model):
        g = p.grad
        if g is None:
            continue
        gf = g.detach()
        # bfloat16 / float16 → float32 に一時変換（CPUへ移して軽量集計）
        gcpu = gf.to(dtype=torch.float32, device="cpu")
        # 集計
        nans = torch.isnan(gcpu).sum().item()
        infs = torch.isinf(gcpu).sum().item()
        zeros = (gcpu == 0).sum().item()
        nan_count += nans
        inf_count += infs
        zero_count += zeros
        total_elems += gcpu.numel()

        ln = torch.linalg.norm(gcpu).item()
        layer_norms.append((ln, name))
        total_sq += ln * ln  # 注意: 厳密な合成ではないが簡易な近似（十分実用）

    # 大域ノルム（近似）
    global_grad_norm = math.sqrt(max(total_sq, 0.0))
    zero_frac = (zero_count / max(total_elems, 1)) if total_elems > 0 else 0.0

    # 上位レイヤを抜粋して記録（大量のレイヤでも軽量）
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
# ESN_mlr network
# =============================================================================
class ESN_mlr(nn.Module):
    """
    Echo-State Network 言語モデル (multi-leak, sparse, low-rank out)。
    学習可能なのは A, B, b_out のみ。
    """

    def __init__(
        self,
        vocab_size: int,
        reservoir_size: int = 1024,
        d: int = 32,                       # 1 列あたり非ゼロ
        spectral_radius: float = 0.99,
        sigma_in: float = 1.0,
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
        activation: str = "tanh",
        dropout: float = 0.1,
        r_out: int = 512,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.reservoir_size = reservoir_size
        self.d = d
        self.gamma = d / self.reservoir_size      # connectivity
        self.device = device
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.sigma_in = sigma_in
        self.spectral_radius = spectral_radius
        self.activation = activation
        self.dropout = dropout
        self.r_out = r_out

        # ---------- leak rate a_i ----------
        a = torch.empty(self.reservoir_size, device=device).uniform_(alpha_min, alpha_max)
        self.register_buffer("a", a)         # shape (N,)

        # ---------- sparse W_in (V × N) ----------
        W_in = self._rand_sparse((self.vocab_size, self.reservoir_size), self.gamma, self.sigma_in, device)
        # 転置して登録しておく (N × V) : sparse × dense 積がしやすい
        self.register_buffer("W_in_T", W_in.transpose(0, 1).coalesce())

        # ---------- sparse W_rec (N × N) ----------
        W_rec = self._rand_sparse((self.reservoir_size, self.reservoir_size), self.gamma, 1.0, device)
        W_rec = self._scale_spectral_radius(W_rec, spectral_radius)
        self.register_buffer("W_rec", W_rec.coalesce())

        # ---------- low-rank output ----------
        self.B = nn.Linear(self.reservoir_size, r_out, bias=False)
        self.A = nn.Linear(r_out, vocab_size, bias=True)
        self.drop = nn.Dropout(dropout)

        # ---------- activation ----------
        if activation == "tanh":
            self.act = torch.tanh
        elif activation == "relu":
            self.act = F.relu
        elif activation == "gelu":
            self.act = F.gelu
        else:
            raise ValueError("activation must be tanh / relu / gelu")

        self.register_buffer("h0", torch.zeros(self.reservoir_size, device=device))
        print(f"ESN_mlr initialized with {self.vocab_size} vocab size, {self.reservoir_size} reservoir size, {self.d} sparsity, {self.gamma} gamma, {self.sigma_in} sigma_in, {self.alpha_min} alpha_min, {self.alpha_max} alpha_max, {self.activation} activation, {self.dropout} dropout, {self.r_out} r_out")
        # 置き換え（ESN_mlr.__init__ の最後あたり）
        a_mean = float(self.a.mean())
        a_std  = float(self.a.std())
        a_minv = float(self.a.min())
        a_maxv = float(self.a.max())
        print(
            f"ESN_mlr init: vocab={self.vocab_size}, N={self.reservoir_size}, d={self.d}, "
            f"gamma={self.gamma:.4f}, sigma_in={self.sigma_in}, "
            f"a∈[{a_minv:.3f},{a_maxv:.3f}] (mean={a_mean:.3f}, std={a_std:.3f}), "
            f"act={self.activation}, dropout={self.dropout}, r_out={self.r_out}"
        )
        # W&B ログ（利用可能時）
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    "a_min": a_minv, "a_max": a_maxv,
                    "a_mean": a_mean, "a_std": a_std
                }, step=0)
        except Exception:
            pass

    # ======================================================================
    # utils
    # ----------------------------------------------------------------------
    @staticmethod
    def _rand_sparse(shape, density, scale, device):
        """shape の COO スパース乱数行列 (N(0,scale^2)) を返す"""
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
        """power iteration で最大固有値を近似し、target_rho にスケール"""
        v = torch.randn(mat.size(0), 1, device=mat.device)
        v /= v.norm() + 1e-9
        for _ in range(iters):
            v = torch.sparse.mm(mat, v)
            v /= v.norm() + 1e-9
        #cur_rho = torch.dot(v.squeeze(), torch.sparse.mm(mat, v).squeeze()).abs().sqrt()
        cur_rho = torch.dot(v.squeeze(), torch.sparse.mm(mat, v).squeeze()).abs()
        mat = mat.coalesce()
        new_vals = mat.values() * (target_rho / (cur_rho + 1e-9))
        return torch.sparse_coo_tensor(mat.indices(), new_vals, mat.size(), device=mat.device)

    # ======================================================================
    # forward
    # ----------------------------------------------------------------------
    def forward(self, x: torch.LongTensor, labels: torch.LongTensor | None = None):
        """
        x: (B, T)
        """
        B, T = x.shape
        h = self.h0.expand(B, -1)            # (B, N)
        a = self.a.unsqueeze(0)              # (1, N)
        outs = []
        for t in range(T):
            # ----- one-hot (B,V) を作成 ------------------------------------------
            one_hot = F.one_hot(x[:, t], num_classes=self.vocab_size).to(h.dtype)  # bf16
            with torch.amp.autocast(device_type="cuda", enabled=False):
                # -- W_in * one_hot -----------------------------------------------
                one_hot_f32 = one_hot.t().float()                  # (V,B) float32
                u_proj = torch.sparse.mm(
                    self.W_in_T.float(),                           # (N,V) float32
                    one_hot_f32                                   # (V,B) float32
                ).t()                                             # → (B,N) float32
                # -- W_rec * h ----------------------------------------------------
                rec = torch.sparse.mm(
                    self.W_rec.float(),                            # (N,N) float32
                    h.float().t()                                 # (N,B) float32
                ).t()                                             # → (B,N) float32
            # bf16 に戻す
            u_proj = u_proj.to(h.dtype)
            rec    = rec.to(h.dtype)
            # ----- state update & output -----------------------------------------
            pre = (u_proj + rec).clamp_(-10.0, 10.0)
            h = (1 - a) * h + a * self.act(pre)
            out = self.A(self.drop(self.B(h)))
            outs.append(out)

        logits = torch.stack(outs, dim=1)         # (B, T, V)

        if labels is None:
            return logits

        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, self.vocab_size),
            labels[:, 1:].reshape(-1),
        )
        return logits, loss
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
    parser.add_argument("--local_batch_size", type=int, default=200, help="Micro batch size per GPU")
    parser.add_argument("--use_gpu_amount", type=int, default=max(torch.cuda.device_count(), 1), help="Total number of GPUs for global batch calculation, if use cpu, set to 1")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument("--validate_every_steps", type=int, default=200, help="Validate every N steps")
    parser.add_argument("--save_checkpoint_every_steps", type=int, default=2000, help="Save checkpoint every N steps")
    parser.add_argument("--generate_every", type=int, default=1000, help="Generate every N steps")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--total_tokens", type=float, default=100e6, help="Total number of tokens to train on")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs (streaming ignored)")
    parser.add_argument("--grad_clip_norm", type=float, default=0, help="Global norm for gradient clipping (<=0 to disable)")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for AdamW")
    parser.add_argument("--beta2", type=float, default=0.95, help="Beta2 for AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay for AdamW")
    parser.add_argument("--tokenizer_path", type=str, default="meta-llama/Llama-2-7b-hf", help="Tokenizer path")


    # ESN
    parser.add_argument("--reservoir_size", type=int, default=1024, help="ESN reservoir size (N_rec)")
    parser.add_argument("--d", type=int, default=2, help="Reservoir sparsity (d/N_rec)")
    parser.add_argument("--spectral_radius", type=float, default=0.95, help="Reservoir spectral radius")
    parser.add_argument("--sigma_in", type=float, default=1.0, help="Input weight init scale")
    parser.add_argument("--alpha_min", type=float, default=0.0, help="Leak rate a_i min")
    parser.add_argument("--alpha_max", type=float, default=1.0, help="Leak rate a_i max")
    parser.add_argument("--activation", choices=["relu","tanh","gelu"], default="tanh", help="Reservoir activation")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout before output")
    parser.add_argument("--r_out", type=int, default=512, help="Output size")

    # Dataset vesteinn/babylm
    parser.add_argument("--dataset_path", type=str, default="HuggingFaceFW/fineweb-edu", help="Dataset path for Training")
    parser.add_argument("--val_dataset_path", type=str, default="vesteinn/babylm", help="Dataset path for Validation")
    # Weights & Biases
    parser.add_argument("--wandb_project", type=str, default="ESN Lanugage Model_N1024", help="WandB project name")
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

def ESN_experiment(lr):
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

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.bos_token = tokenizer.eos_token

    # model
    model = ESN_mlr(
        vocab_size=tokenizer.vocab_size,
        reservoir_size=args.reservoir_size,
        d=args.d,
        spectral_radius=args.spectral_radius,
        sigma_in=args.sigma_in,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        activation=args.activation,
        dropout=args.dropout,
        r_out=args.r_out,
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

    train_loader = get_train_loader(args.tokenizer_path, args.dataset_path, args.seq_len, args.local_batch_size)


    # validation loader unchanged (SlimPajama slice)
    #val_ds = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True)
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
            wandb_run_name = args.wandb_run_name or f"ESN_ml ({param_millions:.2f}M N{args.reservoir_size}_lr{args.learning_rate}_batch_size{args.local_batch_size}_seq_len{args.seq_len}_sigma_in{args.sigma_in}_spectral_radius{args.spectral_radius}_sparsity{1-(args.d/args.reservoir_size)}_dropout{args.dropout}_r_out{args.r_out}_{id}"
            wandb.init(project=args.wandb_project, name=wandb_run_name, config=vars(args))
            WANDB_AVAILABLE = True


    for step, batch in enumerate(train_loader, start=1):
        ids = batch.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, loss = model(ids, labels=ids)

        if distributed:
            model.backward(loss)
            log_gradients_wandb(model, step, tag="preclip")
            log_deepspeed_global_norm_if_available(model, step, tag="preclip")  # 任意

            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.module.parameters(), args.grad_clip_norm)
                log_gradients_wandb(model, step, tag="postclip")
                log_deepspeed_global_norm_if_available(model, step, tag="postclip")  # 任意

            model.step()
            #current_lr = optimizer.param_groups[0]["lr"]
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
            save_dir = f"./checkpoint_1024/ESN_ml ({param_millions:.2f}M N{args.reservoir_size}_batch_size{args.local_batch_size}_seq_len{args.seq_len}_sigma_in{args.sigma_in}_spectral_radius{args.spectral_radius}_sparsity{1-(args.d/args.reservoir_size)}_dropout{args.dropout}_r_out{args.r_out}_{id}"
            os.makedirs(save_dir, exist_ok=True)
            if distributed:
                model.save_checkpoint(save_dir=f"./{save_dir}/{ckpt_name}", tag=f"step_{step}")
            else:
                torch.save(model.state_dict(), f"./{save_dir}/{ckpt_name}")

        if step % args.generate_every == 0 and ((not distributed) or args.local_rank == 0):
            model.eval()
            for prompt in  ["Large Langugage model is "]:
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
                wandb.log({"generated": wandb.Html(f"<b>{prompt}</b>{output_str}")},step=step)
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
            repo_id, api = prepare_hf_repo_id(api_keys["HF_WRITE_TOKEN"], args.hf_repo, default_prefix=f"ESN{param_millions:.2f}M_mlr_N{args.reservoir_size}")
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

        report_path = f"./reports_1024/{report['run_name']}_report.txt"
        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=2))
        print(f"📄 Training report written to {report_path}")
    if WANDB_AVAILABLE:
        wandb.finish()
    del model
    del optimizer
    del scheduler
    del train_loader
    del val_loader
    del val_blocks
    del test_blocks
    del test_ds
    # if WANDB_AVAILABLE:
    #     wandb.finish()


import gc
def mains(lr=1e-3):
    args = parse_args_esn()
    assert 0.0 <= args.alpha_min <= 1.0
    assert 0.0 <  args.alpha_max <= 1.0
    assert args.alpha_min < args.alpha_max

    # ここで実際に学習を実行
    ESN_experiment(lr)
    for i in range(10):
        try:
            ESN_experiment(lr)
        except Exception as e :
            print(e)
            time.sleep(30)
        gc.collect()
        torch.cuda.empty_cache()
if __name__ == "__main__":
    # スクリーニング用の軽量設定（必要に応じて調整）
    a_min_list = [0.00, 0.05, 0.10, 0.20, 0.30]
    a_max_list = [0.50, 0.70, 0.85, 0.95, 1.00]

    import itertools, sys
    combos = [(mi, ma) for mi, ma in itertools.product(a_min_list, a_max_list) if mi < ma - 0.05]
    print(f"[sweep] total combos: {len(combos)} → {combos}")

    # ベース引数を一度だけパース
    base_args = parse_args_esn()

    # 1 設定ずつ実行
    for (mi, ma) in combos:
        # 実行時に a_min / a_max を上書きし、run 名に埋め込む
        if mi==0.00 and ma==0.50 or mi==0.00 and ma==0.70:
            continue
        sys.argv = [
            sys.argv[0],
            "--alpha_min", str(mi),
            "--alpha_max", str(ma),
            "--total_tokens", str(min(base_args.total_tokens, 5e8)),  # 予算圧縮
            "--validate_every_steps", str(base_args.validate_every_steps),
            "--save_checkpoint_every_steps", str(base_args.save_checkpoint_every_steps),
            "--generate_every", str(base_args.generate_every),
            "--local_batch_size", str(base_args.local_batch_size),
            "--seq_len", str(base_args.seq_len),
            "--dataset_path", base_args.dataset_path,
            "--val_dataset_path", base_args.val_dataset_path,
            "--tokenizer_path", base_args.tokenizer_path,
            "--wandb_project", base_args.wandb_project,
            "--wandb_run_name", f"a_min{mi:.2f}_a_max{ma:.2f}"
        ]
        print(f"\n=== Running combo: a_min={mi:.2f}, a_max={ma:.2f} ===")
        mains(lr=base_args.learning_rate)
