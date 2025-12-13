#!/usr/bin/env python3
"""
train template
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
from torch.utils.data import IterableDataset, DataLoader
import torch.distributed as dist

from tqdm.auto import tqdm
import wandb
from datasets import load_dataset
from transformers import LlamaTokenizer, get_cosine_schedule_with_warmup
from huggingface_hub import HfApi, upload_folder


try:
    import deepspeed
    _HAS_DS = True
except ImportError:
    _HAS_DS = False

# ---------- Constants ----------
MAGIC_NUMBER = 20240520
HEADER_INT32  = 256  # 1 KiB header = 256 * int32
VERSION = 1
HEADER_U16 = HEADER_INT32 *2
class BinShardsDataset(IterableDataset):
    """Load pre‑tokenised uint16 shards with zero‑copy mmap."""

    def __init__(self, shard_dir: str | Path, seq_len: int):
        self.seq_len = seq_len
        self.files = sorted(Path(shard_dir).glob("shard_*.bin"))
        if not self.files:
            raise FileNotFoundError(f"no shard_*.bin in {shard_dir}")
        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
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

    def __iter__(self):
        for i, f in enumerate(self.files):
            if i % self.world_size != self.rank:
                continue
            toks = self._load_uint16_tokens(f)
            for j in range(0, len(toks) - self.seq_len + 1, self.seq_len):
                yield toks[j : j + self.seq_len].long()




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


def prepare_hf_repo_id(write_token, explicit_repo, default_prefix="gpt2-scratch"):
    """Helper to build repo id if not explicitly given."""
    api = HfApi(token=write_token)
    if explicit_repo:
        return explicit_repo, api
    me = api.whoami()
    owner = me["name"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    repo_id = f"{owner}/{default_prefix}-{timestamp}"
    return repo_id, api

def rotate_every_two(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)

def apply_rotary_pos_emb(x, cos, sin):
    # x: (B, n_head, T, head_dim)
    return (x * cos) + (rotate_every_two(x) * sin)

@torch.no_grad()
def sample_sequence(
    model,
    input_ids: torch.LongTensor,
    max_new_tokens: int = 20,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    min_p: float = 0.0,
    device: torch.device = None,
):
    """Greedy/Top-k/Top-p sampling util."""
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


class StreamingDataset(IterableDataset):
    def __init__(self, dataset_path, split, tokenizer, seq_len):
        self.ds = load_dataset(
            dataset_path,
            split=split,
            streaming=True).shuffle(buffer_size=10_000_000,seed=42)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        # distributed info
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

    def __iter__(self):
        buf = []
        for idx, ex in enumerate(self.ds):
            if idx % self.world_size != self.rank:
                continue
            toks = self.tokenizer(ex["text"], return_attention_mask=False)["input_ids"]
            buf.extend(toks)
            while len(buf) >= self.seq_len:
                yield torch.tensor(buf[: self.seq_len], dtype=torch.long)
                buf = buf[self.seq_len :]

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

# ---------- Data loader helper ----------

def get_train_loader(path: str, seq_len: int, batch_size: int):
    if Path(path).is_dir():
        ds = BinShardsDataset(path, seq_len)
        print(f"[data] BinShardsDataset with {len(list(Path(path).glob('shard_*.bin')))} files")
        tot = len(list(Path(path).glob('shard_*.bin')))
        per_rank = (tot + ds.world_size - 1) // ds.world_size
        print(f"[data] BinShardsDataset: {tot} shards → {per_rank} / rank (world={ds.world_size})")

    else:
        tok = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True)
        ds = StreamingDataset(path, "train", tok, seq_len)  # original tokenizer‑on‑the‑fly dataset
        print("[data] HF streaming dataset", path)
    return DataLoader(ds, batch_size=batch_size, num_workers=4 if Path(path).is_dir() else 8,
                      pin_memory=True, drop_last=True)


# ---------- Evaluation ----------
# @torch.no_grad()
# def compute_mean_so_far_ppl(model, blocks, device, ks):
#     """
#     model   : Model or deepspeed モデル
#     blocks  : list of {"input_ids": Tensor[T], "labels": Tensor[T]}
#     device  : torch.device
#     ks      : list of int (計測したいトークン長のリスト)
#     returns : {k: {"mean_nll": float, "mean_ppl": float}}
#     """
#     sum_logprob = {k: 0.0 for k in ks}
#     token_count = {k: 0 for k in ks}

#     model.eval()
#     for block in blocks:
#         ids    = block["input_ids"].to(device).unsqueeze(0)   # (1, T)
#         labels = block["labels"].to(device).unsqueeze(0)     # (1, T)   
#         out =model(ids)
#         logits = out[0] if isinstance(out, tuple) else out                      # (1, T, V)
#         log_probs = F.log_softmax(logits, dim=-1)            # (1, T, V)

#         # shift して位置 i の log P を取得
#         #  predict for token i → log_probs[0, i-1, label_i]
#         lp = log_probs[0, :-1, :]                            # predict positions 1..T-1
#         lbl = labels[0, 1:]                                  # true tokens at 1..T-1
#         lp_i = lp.gather(1, lbl.unsqueeze(1)).squeeze(1)     # (T-1,) vector

#         T = lp_i.size(0)
#         for k in ks:
#             k_trunc = min(k, T)
#             sum_logprob[k] += lp_i[:k_trunc].sum().item()
#             token_count[k] += k_trunc

#     # 平均 NLL → PPL に
#     results = {}
#     for k in ks:
#         mean_nll = - sum_logprob[k] / token_count[k]
#         results[k] = {
#             "mean_nll": mean_nll,
#             "mean_ppl": math.exp(mean_nll)
#         }
#     return results

@torch.no_grad()
def compute_mean_so_far_ppl(model, blocks, device, ks):
    results = {k: [] for k in ks}
    for block in tqdm(blocks, desc="Computing mean-so-far PPL"):
        ids = block["input_ids"].to(device).unsqueeze(0)   # (1, T)
        logits, _ = model(ids, labels=ids)                # (1, T, V), loss を無視
        # 予測分は shift
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (1, T-1, V)
        # 真ラベルの log_prob を取る
        true_lp = torch.gather(log_probs, 2, ids[:,1:].unsqueeze(-1)).squeeze(-1)  # (1, T-1)
        true_lp = true_lp.squeeze(0)  # (T-1,)

        # GPU 上で累積和 & 平均化
        cum_lp = torch.cumsum(true_lp, dim=0)  # (T-1,)
        lengths = torch.arange(1, cum_lp.size(0) + 1, device=device)  # (1,2,...,T-1)
        mean_nll = - cum_lp / lengths        # (T-1,)
        # ks の位置だけピックアップ
        for k in ks:
            k_idx = min(k, mean_nll.size(0)) - 1
            results[k].append(mean_nll[k_idx].exp().item())  # PPL = exp(mean_nll)

    # 最後に各 k ごとに平均を取る
    return {k: sum(results[k]) / len(results[k]) for k in ks}


# ---------- ESN+i モデル ----------
import torch
import torch.nn as nn
import torch.nn.functional as F

class ESNi(nn.Module):
    """
    Echo State Network with trainable embedding (ESN+i) implemented in PyTorch.
    Dense implementation (no sparse ops), fully compatible with mixed-precision (bfloat16) training.

    Args:
        vocab_size (int): Size of the vocabulary (number of one-hot inputs).
        reservoir_size (int): Number of reservoir neurons (N_rec).
        leaking_rate (float): Leak rate a in (0, 1].
        activation (str): 'relu' or 'tanh'.
        sigma_in (float): Std for input weight init.
        spectral_radius (float): Desired spectral radius for W_rec.
        reservoir_sparsity (float): Fraction zeros in W_rec mask.
        train_embedding (bool): If True, embedding layer is trainable (ESN+i); else fixed dense W_in.
        dropout (float): Dropout probability before output.
        device (torch.device): Device for initial buffers (but model should be .to(device) after init).
    """
    def __init__(
        self,
        vocab_size: int,
        reservoir_size: int = 512,
        leaking_rate: float = 0.8,
        activation: str = 'relu',
        sigma_in: float = 1.0,
        spectral_radius: float = 0.993,
        reservoir_sparsity: float = 0.5,
        train_embedding: bool = True,
        dropout: float = 0.1,
        device: torch.device = torch.device('cpu')
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.Nrec = reservoir_size
        self.a = leaking_rate

        # Activation function
        if activation == 'relu':
            self.act = F.relu
        elif activation == 'tanh':
            self.act = torch.tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Input projection: embedding or fixed dense W_in
        if train_embedding:
            self.embedding = nn.Embedding(vocab_size, reservoir_size)
            nn.init.normal_(self.embedding.weight, mean=0.0, std=sigma_in)
            self.W_in = None
        else:
            self.embedding = None
            W_in = torch.randn(vocab_size, reservoir_size, device=device) * sigma_in
            self.W_in = nn.Parameter(W_in, requires_grad=False)

        # Reservoir weights: fixed dense W_rec with spectral radius scaling
        W_rand = torch.randn(reservoir_size, reservoir_size, device=device)
        mask = (torch.rand(reservoir_size, reservoir_size, device=device) > reservoir_sparsity).float()
        W_masked = W_rand * mask
        def compute_spectral_radius(mat, num_iters=20):
            x = torch.randn(mat.size(0), 1, device=mat.device)
            for _ in range(num_iters):
                x = mat @ x
                x = x / (x.norm() + 1e-9)
            return (x.t() @ (mat @ x)).abs().sqrt().item()
        rho_est = compute_spectral_radius(W_masked)
        W_scaled = W_masked * (spectral_radius / (rho_est + 1e-9))
        self.W_rec = nn.Parameter(W_scaled, requires_grad=False)

        # Output layer
        self.out_layer = nn.Linear(reservoir_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

        # Initial reservoir state buffer
        self.register_buffer('h0', torch.zeros(1, reservoir_size, device=device))
    #@torch.jit.script
    def forward(
        self,
        x: torch.LongTensor,
        labels: torch.LongTensor = None
    ):
        """
        Args:
            x: LongTensor of shape (batch_size, seq_len) with token indices.
            labels: LongTensor (batch_size, seq_len) if provided for LM loss.
        Returns:
            logits: (batch_size, seq_len, vocab_size)
            loss: Tensor if labels provided, otherwise omitted.
        """
        B, T = x.size()
        h = self.h0.expand(B, -1).clone()
        outputs = []
        for t in range(T):
            u_t = x[:, t]
            if self.embedding is not None:
                u_proj = self.embedding(u_t)
            else:
                one_hot = F.one_hot(u_t, num_classes=self.vocab_size).to(dtype = h.dtype)
                u_proj = one_hot @ self.W_in
            rec = h @ self.W_rec
            pre_act = rec + u_proj
            h = (1 - self.a) * h + self.a * self.act(pre_act)
            o = self.out_layer(self.dropout(h))
            outputs.append(o)
        logits = torch.stack(outputs, dim=1)
        if labels is not None:
            shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
            shift_labels = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            return logits, loss
        return logits

# ---------- main() skeleton ----------
def parse_args():
    parser = argparse.ArgumentParser(description="GPT2 scratch pretrain with optional DeepSpeed and CLI-configurable parameters")
    # Distributed and DeepSpeed options
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)), help="Local rank for distributed training")
    parser.add_argument("--deepspeed_config", type=str, default="ds_config.json", help="Path to DeepSpeed config file")
    parser.add_argument("--use_deepspeed", action="store_true", help="Enable DeepSpeed if available and GPUs > 1")

    # Training parameters
    #parser.add_argument("--local_batch_size", type=int, default=4, help="Micro batch size per GPU")
    parser.add_argument("--local_batch_size", type=int, default=64, help="Micro batch size per GPU")
    parser.add_argument("--use_gpu_amount", type=int, default=max(torch.cuda.device_count(), 1), help="Total number of GPUs for global batch calculation, if use cpu, set to 1")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate")
    parser.add_argument("--validate_every_steps", type=int, default=200, help="Validate every N steps")
    parser.add_argument("--save_checkpoint_every_steps", type=int, default=2000, help="Save checkpoint every N steps")
    parser.add_argument("--generate_every", type=int, default=1000, help="Generate every N steps")
    parser.add_argument("--seq_len", type=int, default=256, help="Sequence length")
    parser.add_argument("--total_tokens", type=float, default=17e7, help="Total number of tokens to train on")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs (streaming ignored)")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="Global norm for gradient clipping (<=0 to disable)")

    # ESN+i モデル用パラメータ
    parser.add_argument("--reservoir_size", type=int, default=550, help="ESN reservoir size (N_rec)")
    parser.add_argument("--leaking_rate", type=float, default=0.8, help="Leaking rate a ∈ (0,1]")
    parser.add_argument("--activation", choices=["relu","tanh"], default="tanh", help="Reservoir activation")
    parser.add_argument("--sigma_in", type=float, default=1.0, help="Input weight init scale")
    parser.add_argument("--spectral_radius", type=float, default=0.993, help="Reservoir spectral radius")
    parser.add_argument("--reservoir_sparsity", type=float, default=0.5, help="Fraction of zeros in W_rec")
    parser.add_argument("--train_embedding", action="store_true", help="Embed 層を学習可能にする (ESN+i)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout before output")
    # Dataset
    parser.add_argument("--dataset_path", type=str, default="vesteinn/babylm", help="Dataset path for Training")
    parser.add_argument("--val_dataset_path", type=str, default="vesteinn/babylm", help="Dataset path for Validation")
    # Weights & Biases
    parser.add_argument("--wandb_project", type=str, default="ESN_LM", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name (default: generated by WandB)")
    parser.add_argument("--api_file", type=str, default="api.txt", help="API file path")

    parser.add_argument("--hf_repo", type=str, default=None, help="Upload destination like 'username/gpt2-scratch-64M'. If None, skip upload")
    parser.add_argument("--hf_private", action="store_true", help="Create HF repo as private")

    return parser.parse_args()

def main():
    args = parse_args()
    WANDB_AVAILABLE = False
    api_keys = load_api_keys(args.api_file)
    if "WANDB_API_KEY" in api_keys:
        os.environ["WANDB_API_KEY"] = api_keys["WANDB_API_KEY"]
    if "HF_READ_TOKEN" in api_keys:
        from huggingface_hub import login

        login(token=api_keys["HF_WRITE_TOKEN"])
    if args.use_deepspeed and _HAS_DS and torch.cuda.device_count() > 1:
        torch.cuda.set_device(args.local_rank)
        distributed = True
    else:
        distributed = False

    if args.use_deepspeed and _HAS_DS and torch.cuda.device_count() > 1:
        torch.cuda.set_device(args.local_rank)
        distributed = True
    else:
        distributed = False
    world_size = dist.get_world_size() if (distributed and dist.is_initialized*()) else 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokenizer (needed for val & generation)
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = ESNi(
        vocab_size=tokenizer.vocab_size,
        reservoir_size=args.reservoir_size,
        leaking_rate=args.leaking_rate,
        activation=args.activation,
        sigma_in=args.sigma_in,
        spectral_radius=args.spectral_radius,
        reservoir_sparsity=args.reservoir_sparsity,
        train_embedding=args.train_embedding,
        dropout=args.dropout,
        device=device,
    )
    model.to(device)
    model.to(torch.bfloat16)
    print(f"parameter count: {sum(p.numel() for p in model.parameters())}")


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
            lr=args.learning_rate,
            weight_decay=0.1,
        )
        total_steps = math.ceil(args.total_tokens / (args.local_batch_size * args.use_gpu_amount * args.seq_len))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=math.ceil(0.1 * total_steps),
            num_training_steps=total_steps,
        )
        model.to(device)

    train_loader = get_train_loader(args.dataset_path, args.seq_len, args.local_batch_size)


    # validation loader unchanged (SlimPajama slice)
    #val_ds = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True)
    val_ds = load_dataset(args.val_dataset_path, split="test", streaming=False)
    val_blocks = get_validation_blocks(list(islice(val_ds, 100)), tokenizer, args.seq_len, 1)
    val_loader = DataLoader(val_blocks, batch_size=args.local_batch_size, num_workers=0, pin_memory=True)

    max_steps = math.ceil(args.total_tokens / (args.local_batch_size * args.use_gpu_amount * args.seq_len))

    tokens_seen_local = 0
    tokens_seen_global = 0

    start_time = time.time()
    max_mem_mb = 0.0
    model.train()
    if not distributed or args.local_rank == 0:
            wandb_run_name = args.wandb_run_name or f"{args.wandb_project}_{args.reservoir_size}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(project=args.wandb_project, name=wandb_run_name, config=vars(args))
            WANDB_AVAILABLE = True


    for step, batch in enumerate(train_loader, start=1):
        ids = batch.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, loss = model(ids, labels=ids)
        if torch.isnan(loss):
            print("⚠️ NaN loss on step", step)
            print("  logits stats:", logits.mean().item(), logits.std().item())
            print("  labels:", ids[0,:10])
            break

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
            save_dir = f"./checkpoint/{args.wandb_project}_{start_time}"
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
            model.train()
        # if step >= max_steps:
        #     break
        if tokens_seen_global >= args.total_tokens:
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
        if "HF_WRITE_TOKEN" in api_keys:
            repo_id, api = prepare_hf_repo_id(api_keys["HF_WRITE_TOKEN"], args.hf_repo)
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
        }

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
        test_ds = load_dataset(args.dataset_path, split="test", streaming=False)
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
        print(f"→ {len(test_blocks)} ブロックを作成")
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

        report["test_mean_so_far_ppl_curve"] = {
            k: mean_so_far[k]["mean_ppl"] for k in ks
        }

        report_path = f"./{report['run_name']}_report.txt"
        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=2))
        print(f"📄 Training report written to {report_path}")

    if WANDB_AVAILABLE:
        wandb.finish()

if __name__ == "__main__":
    main()

