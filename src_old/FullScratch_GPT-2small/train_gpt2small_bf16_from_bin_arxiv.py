#!/usr/bin/env python3
"""
train_gpt2_small_bin.py
=======================
GPT‑2 small を **事前トークナイズ済み uint16 シャード** から高速学習するスクリプト。
- 元の `train_gpt2_small.py` からの主な変更点
  * `BinShardsDataset` : `/path/to/shards/` 以下の `shard_*.bin` を 0‑copy `torch.from_file` で読み込み
  * `dataset_path` にディレクトリを渡すとシャードモード、HF データセット名を渡すと従来のストリーミングモード
  * バリデーションは従来どおり SlimPajama 先頭 100 サンプルをトークナイズ

使い方例
---------
```bash
# 1GPU で bfloat16 DeepSpeed なしスクリプトを直接実行
python train_gpt2_small_bin.py \
    --dataset_path /nvme/fineweb_tokens \
    --seq_len 2048 \
    --local_batch_size 32 \
    --total_tokens 15e9 \
    --use_gpu_amount 1 \
    --wandb_project gpt2-scratch-pretrain \
    --api_file api.txt

# DeepSpeed で
deepspeed --num_gpus 1 train_gpt2_small_bin.py \
    --dataset_path /nvme/fineweb_tokens \
    --use_deepspeed --deepspeed_config ds_config_bf16.json \
    --local_batch_size 32 --total_tokens 15e9 \
    --wandb_project gpt2-scratch-pretrain --api_file api.txt
```
"""
import argparse
import os
import math
import datetime
import time
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
import wandb
from datasets import load_dataset
from transformers import LlamaTokenizer, get_cosine_schedule_with_warmup
from huggingface_hub import HfApi, upload_folder
from itertools import islice
import torch.distributed as dist


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
# ---------- Bin‑shard dataset ----------
class BinShardsDataset(IterableDataset):
    #def __init__(self, shards_dir: str | Path, seq_len: int):
        # self.files   = sorted(Path(shards_dir).glob("shard_*.bin"))
        # if not self.files:
        #     raise FileNotFoundError(f"no shard_*.bin in {shards_dir}")
        # self.seq_len = seq_len

    def __init__(self,shard_dir:str | Path, seq_len:int):
        self.seq_len = seq_len

        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

    def _load_uint16_tokens(self, file: Path) -> torch.Tensor:
        # ① ヘッダだけ mmap
        header = torch.from_file(str(file),
                                 dtype=torch.int32,
                                 size = HEADER_INT32)
        if header[0].item() != MAGIC_NUMBER or header[1].item() != VERSION:
            raise ValueError(f"bad header in {file}")
        num_tok = int(header[2].item())

        # ② ヘッダ + 本体を uint16 で mmap
        tot_u16 = HEADER_U16 + num_tok          # 必要な uint16 要素数
        mapped  = torch.from_file(str(file),
                                  dtype=torch.uint16,
                                  size = tot_u16)

        # ③ ヘッダぶんをスライスで捨てる（コピーなし）
        return mapped[HEADER_U16:]              # 長さ = num_tok

    def __iter__(self):
        for i, f in enumerate(self.files):
            #rankごとのファイル割り当て
            if  i % self.world_size != self.rank:
                continue
            toks = self.__load
            for i in range(0,len(toks) -self.seq_len + 1,self.seq_len ):
                yield toks[i : i+ self.seq_len]

def parse_args():
    parser = argparse.ArgumentParser(description="GPT2 scratch pretrain with optional DeepSpeed and CLI-configurable parameters")
    # Distributed and DeepSpeed options
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)), help="Local rank for distributed training")
    parser.add_argument("--deepspeed_config", type=str, default="ds_config.json", help="Path to DeepSpeed config file")
    parser.add_argument("--use_deepspeed", action="store_true", help="Enable DeepSpeed if available and GPUs > 1")

    # Training parameters
    parser.add_argument("--local_batch_size", type=int, default=4, help="Micro batch size per GPU")
    parser.add_argument("--use_gpu_amount", type=int, default=max(torch.cuda.device_count(), 1), help="Total number of GPUs for global batch calculation, if use cpu, set to 1")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate")
    parser.add_argument("--validate_every_steps", type=int, default=200, help="Validate every N steps")
    parser.add_argument("--save_checkpoint_every_steps", type=int, default=2000, help="Save checkpoint every N steps")
    parser.add_argument("--generate_every", type=int, default=1000, help="Generate every N steps")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--total_tokens", type=float, default=15e9, help="Total number of tokens to train on")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs (streaming ignored)")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="Global norm for gradient clipping (<=0 to disable)")

    # Model parameters
    parser.add_argument("--n_embd", type=int, default=768, help="Embedding dimension size")
    parser.add_argument("--n_layer", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=12, help="Number of attention heads per layer")

    # Dataset
    parser.add_argument("--dataset_path", type=str, default="HuggingFaceFW/fineweb-edu", help="Dataset path for Training")

    # Weights & Biases
    parser.add_argument("--wandb_project", type=str, default="gpt2-scratch-pretrain", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name (default: generated by WandB)")
    parser.add_argument("--api_file", type=str, default="api.txt", help="API file path")

    parser.add_argument("--hf_repo", type=str, default=None, help="Upload destination like 'username/gpt2-scratch'. If None, skip upload")
    parser.add_argument("--hf_private", action="store_true", help="Create HF repo as private")

    return parser.parse_args()

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

# ---------- GPT‑2 model definitions ----------

class GPT2Config:
    def __init__(self, vocab_size, n_positions, n_embd, n_layer, n_head):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_ctx = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.register_buffer("mask", torch.tril(torch.ones(config.n_ctx, config.n_ctx)).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        y = att @ v
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT2LMHeadModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, input_ids, labels=None):
        B, T = input_ids.size()
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.wte(input_ids) + self.wpe(pos)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)).float(), shift_labels.view(-1))
        return logits, loss


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
        logits = outputs[0]
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
        self.ds = load_dataset(dataset_path, split=split, streaming=True)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        buf = []
        for ex in self.ds:
            toks = self.tokenizer(ex["text"], return_attention_mask=False)["input_ids"]
            buf.extend(toks)
            while len(buf) >= self.seq_len:
                yield torch.tensor(buf[: self.seq_len], dtype=torch.long)
                buf = buf[self.seq_len :]

def get_validation_blocks(hf_dataset, tokenizer, seq_len, max_blocks=50):
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

# ---------- main() skeleton ----------

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokenizer (needed for val & generation)
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # model
    config = GPT2Config(
        vocab_size = tokenizer.vocab_size,
        n_positions = args.seq_len,
        n_embd = args.n_embd,
        n_layer = args.n_layer,
        n_head = args.n_head,
    )
    model = GPT2LMHeadModel(config).to(torch.bfloat16)

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
    slim = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True)
    val_blocks = get_validation_blocks(list(islice(slim, 100)), tokenizer, args.seq_len, 1)
    val_loader = DataLoader(val_blocks, batch_size=args.local_batch_size, num_workers=0, pin_memory=True)

    max_steps = math.ceil(args.total_tokens / (args.local_batch_size * args.use_gpu_amount * args.seq_len))

    tokens_seen_local = 0
    tokens_seen_global = 0

    start_time = time.time()
    max_mem_mb = 0.0
    model.train()

    if not distributed or args.local_rank == 0:
            wandb_run_name = args.wandb_run_name or f"{args.wandb_project}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(project=args.wandb_project, name=wandb_run_name, config=vars(args))
            WANDB_AVAILABLE = True


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
        tokens_seen_global += args.local_batch_size * args.seq_len * dist.get_world_size()
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
            ckpt_name = f"checkpoint_{step}_{tokens_seen_global}_tokens.pt"
            if distributed:
                model.save_checkpoint(save_dir=".", tag=f"step_{step}")
            else:
                torch.save(model.state_dict(), ckpt_name)

        if step % args.generate_every == 0 and ((not distributed) or args.local_rank == 0):
            model.eval()
            for prompt in ["こんにちは，", "4 + 11 = "]:
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
            json.dump(config.__dict__, f, indent=2)

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
        test_input = torch.randint(0, tokenizer.vocab_size, (1, 1024), device=device)
        torch.cuda.reset_peak_memory_stats()
        t_inf_start = time.time()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _ = model(test_input)
        infer_time = time.time() - t_inf_start
        infer_tok_per_sec = 1024 / infer_time if infer_time > 0 else 0.0
        infer_mem_mb = torch.cuda.max_memory_allocated() / 1e6
        report.update({"inference_tok_per_sec": infer_tok_per_sec, "inference_mem_MB": infer_mem_mb})

        report_path = f"{report['run_name']}_report.txt"
        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=2))
        print(f"📄 Training report written to {report_path}")

    if WANDB_AVAILABLE:
        wandb.finish()

if __name__ == "__main__":
    main()

