#!/usr/bin/env python3
"""
fast_tokenize.py 
======================================
Pre‑tokenize large HF dataset → fixed‑length uint16 shard files compatible with RTA trainer.

* `--resume` で既存 shard 数を検出し、その数 × `shard_size` 件だけ
  - `datasets >= 2.18` なら `IterableDataset.skip(N)` を使用（O(1)）
  - それ未満ならシーケンスを **素通し読み** で高速に読み飛ばす（O(N) だがトークナイズしないので数十秒）
* tqdm 進捗と（API キーがあれば）WandB ログ。
"""
import os, uuid, multiprocessing as mp
from pathlib import Path
from itertools import islice
import tqdm, torch, wandb, datasets
from datasets import load_dataset
from transformers import LlamaTokenizer

# ---------- constants ----------
MAGIC_NUMBER = 20251118
VERSION       = 1
HEADER_INT32  = 256  # 1 KiB header
MODEL_NAME    = "meta-llama/Llama-2-7b-hf"

# ---------- argument parser ----------

def parse_args():
    import argparse
    p = argparse.ArgumentParser("Pre-tokenize dataset → uint16 shards (resume supported)")
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--shard_size", type=int, default=10_000)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--wandb_project", default="tokenize-prep")
    p.add_argument("--resume", action="store_true", help="continue after last existing shard")
    return p.parse_args()

# ---------- worker helpers ----------

global_tok = None

def _init_tokenizer():
    global global_tok
    global_tok = LlamaTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    global_tok.pad_token = global_tok.eos_token

def _encode(texts):
    """Encode list[str] → list[list[int]] (per process)"""
    return global_tok(texts, add_special_tokens=False, padding=False).input_ids

# ---------- shard writer ----------

def write_shard(seq_tensors, idx: int, out_dir: Path):
    data = torch.stack(seq_tensors).flatten().to(torch.uint16)
    header = torch.zeros(HEADER_INT32, dtype=torch.int32)
    header[:3] = torch.tensor([MAGIC_NUMBER, VERSION, data.numel()], dtype=torch.int32)
    out_dir.mkdir(parents=True, exist_ok=True)
    file = out_dir / f"shard_{idx:05d}.bin"
    with file.open("wb") as f:
        f.write(header.numpy())
        f.write(data.numpy())
    print(f"✔ shard {idx:05d}: {len(seq_tensors)} seq → {file}")

# ---------- main ----------

def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    existing = sorted(out_dir.glob("shard_*.bin")) if args.resume and out_dir.exists() else []
    start_shard = max(int(f.stem.split('_')[1]) for f in existing) + 1 if existing else 0
    seq_to_skip = start_shard * args.shard_size
    print(f"[resume] start shard {start_shard:05d} (skip {seq_to_skip:,} sequences)")

    # --- WandB ---
    use_wb = bool(os.getenv("WANDB_API_KEY"))
    if use_wb:
        wb_run = wandb.init(project=args.wandb_project, name=f"tok_{uuid.uuid4().hex[:6]}")
        wandb.config.update(vars(args))

    # --- Dataset iterator with optional skip ---
    raw_ds = load_dataset(args.dataset_path, split=args.split, streaming=True)

    ver_minor = int(datasets.__version__.split('.')[1])
    native_skip_ok = ver_minor >= 18 and hasattr(raw_ds, "skip")

    if seq_to_skip:
        if native_skip_ok:
            raw_ds = raw_ds.skip(seq_to_skip)
            ds_iter = iter(raw_ds)
        else:
            print("[resume] datasets < 2.18 → manual fast-forward", flush=True)
            ds_iter = iter(raw_ds)
            for _ in islice(ds_iter, seq_to_skip):
                pass  # fast consume without tokenization
            print(f"[resume] manually consumed {seq_to_skip:,} sequences", flush=True)
    else:
        ds_iter = iter(raw_ds)

    # --- Multiprocessing pool ---
    pool = mp.Pool(args.num_workers, initializer=_init_tokenizer) if args.num_workers else None
    encode = _encode if pool is None else pool.map

    # --- Buffers & counters ---
    seq_buf, shard_buf = [], []
    shard_idx, total_seq, total_tok = start_shard, 0, 0

    pbar = tqdm.tqdm(desc="texts", unit="sample")
    while True:
        raw_batch = list(islice(ds_iter, args.batch_size))
        if not raw_batch:
            break
        texts = [ex.get("text", "") for ex in raw_batch if ex.get("text", "")]
        if not texts:
            continue

        id_lists = [encode(texts)] if pool is None else pool.map(_encode, [texts])
        for ids in id_lists[0]:
            seq_buf.extend(ids)
            while len(seq_buf) >= args.seq_len:
                shard_buf.append(torch.tensor(seq_buf[:args.seq_len], dtype=torch.uint16))
                seq_buf = seq_buf[args.seq_len:]
                total_seq += 1; total_tok += args.seq_len
                if len(shard_buf) >= args.shard_size:
                    write_shard(shard_buf, shard_idx, out_dir)
                    shard_buf.clear(); shard_idx += 1
        pbar.update(len(texts))
        if use_wb and total_seq % (args.shard_size * 2) == 0:
            wandb.log({"seq": total_seq, "tok": total_tok, "shard": shard_idx})

    # write remainder
    if shard_buf:
        write_shard(shard_buf, shard_idx, out_dir)

    # cleanup
    pbar.close()
    if pool:
        pool.close(); pool.join()
    if use_wb:
        wandb.log({"seq": total_seq, "tok": total_tok, "shard": shard_idx, "status": "done"});
        wb_run.finish()
    print(f"Done. {total_seq:,} sequences → {shard_idx+1} shards @ {out_dir}")


if __name__ == "__main__":
    main()

