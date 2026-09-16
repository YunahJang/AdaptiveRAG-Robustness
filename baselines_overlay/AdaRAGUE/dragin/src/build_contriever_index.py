# Part of the QueryRobustness overlay for AdaRAGUE (https://github.com/s-nlp/AdaRAGUE).
# MIT License, Copyright (c) 2025 Natural Language Processing.
# Added by this project.

import os as _os
# Project root. Override with:  export QR_ROOT=/path/to/QueryRobustness
QR_ROOT = _os.environ.get(
    "QR_ROOT",
    _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "..")),
)

import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel
import faiss

# =========================================================
# Settings (the only block you need to adapt to your environment)
# =========================================================
TSV_PATH = f"{QR_ROOT}/datasets/data/dpr/psgs_w100.tsv"
OUT_DIR  = f"{QR_ROOT}/indexes/contriever_psgs_w100_fp16_stream"

MODEL_NAME = "facebook/contriever"

# Speed / accuracy trade-off:
# - lowering this to 128 is usually much faster
MAX_LENGTH = 256

# Raise as far as GPU memory allows (512-4096 is typical)
BATCH_SIZE = 4096

# TSV streaming chunk size (RAM vs. speed)
CHUNK_ROWS = 200_000

# whether to prepend the title to the passage
USE_TITLE = True

# debugging only (-1 = all rows)
LIMIT_ROWS = -1

# resume a previous, interrupted run
RESUME = True
# =========================================================


def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B,T,1]
    summed = (last_hidden_state * mask).sum(dim=1)                  # [B,H]
    counts = mask.sum(dim=1).clamp(min=1e-9)                        # [B,1]
    return summed / counts


@torch.no_grad()
def encode_batch(texts, tokenizer, model, device, max_length, use_fp16):
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch = {k: v.to(device) for k, v in batch.items()}

    if use_fp16:
        # fp16 autocast (works well on Ampere and newer)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = model(**batch)
    else:
        out = model(**batch)

    emb = mean_pooling(out.last_hidden_state, batch["attention_mask"])
    emb = emb.detach().float().cpu().numpy().astype("float32")  # FAISS expects float32
    faiss.normalize_L2(emb)  # for cosine similarity
    return emb


def append_jsonl(path, records):
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_processed_count(progress_path: str) -> int:
    if not os.path.exists(progress_path):
        return 0
    with open(progress_path, "r", encoding="utf-8") as f:
        s = f.read().strip()
    try:
        return int(s)
    except:
        return 0


def write_processed_count(progress_path: str, n: int):
    tmp = progress_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(str(n))
    os.replace(tmp, progress_path)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = (device == "cuda")

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_path = str(out_dir / "index.faiss")
    meta_path = str(out_dir / "meta.jsonl")
    progress_path = str(out_dir / "progress.txt")

    print("========================================")
    print("🚀 Build Contriever FAISS Index (stream + fp16)")
    print("========================================")
    print(f"TSV_PATH   : {TSV_PATH}")
    print(f"OUT_DIR    : {OUT_DIR}")
    print(f"DEVICE     : {device}")
    print(f"FP16       : {use_fp16}")
    print(f"MAX_LENGTH : {MAX_LENGTH}")
    print(f"BATCH_SIZE : {BATCH_SIZE}")
    print(f"CHUNK_ROWS : {CHUNK_ROWS}")
    print(f"USE_TITLE  : {USE_TITLE}")
    print(f"LIMIT_ROWS : {LIMIT_ROWS}")
    print(f"RESUME     : {RESUME}")
    print("========================================\n")

    # Resume: continue when an index/meta pair already exists
    processed = 0
    if RESUME and os.path.exists(index_path) and os.path.exists(meta_path):
        processed = read_processed_count(progress_path)
        print(f"[RESUME] Found existing index/meta. processed_rows={processed}")
        index = faiss.read_index(index_path)
        # meta is appended to, so it is not loaded here
    else:
        if os.path.exists(index_path) or os.path.exists(meta_path):
            print("[WARN] Existing files found but RESUME disabled or incomplete. Overwriting.")
        # start from scratch
        if os.path.exists(index_path): os.remove(index_path)
        if os.path.exists(meta_path): os.remove(meta_path)
        if os.path.exists(progress_path): os.remove(progress_path)
        index = None

    print("[1] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    # use the first batch to get the dimension and create the index (fresh runs only)
    if index is None:
        # read a single chunk of the TSV to obtain one sample
        df0 = next(pd.read_csv(TSV_PATH, sep="\t", dtype=str, chunksize=1))
        t0 = (df0["title"].iloc[0] if "title" in df0.columns else "") or ""
        x0 = (df0["text"].iloc[0] if "text" in df0.columns else "") or ""
        sample = f"{t0.strip()} [SEP] {x0.strip()}" if (USE_TITLE and t0.strip()) else x0.strip()
        emb0 = encode_batch([sample], tokenizer, model, device, MAX_LENGTH, use_fp16)
        dim = emb0.shape[1]
        print(f"[INFO] embedding dim = {dim}")
        index = faiss.IndexFlatIP(dim)  # cosine = normalize + IP
        # fresh run: start from an empty meta file
        open(meta_path, "w", encoding="utf-8").close()

    print("[2] Streaming TSV and building index...")
    # skiprows can be slow in pandas, so we read chunk by chunk
    # and discard the rows that were already processed.
    rows_seen = 0
    rows_added = 0

    t_start = time.time()

    # tqdm total: dynamic when unknown, LIMIT_ROWS when set
    pbar_total = None if LIMIT_ROWS < 0 else max(LIMIT_ROWS - processed, 0)
    pbar = tqdm(total=pbar_total, desc="passages", unit="rows")

    for chunk in pd.read_csv(TSV_PATH, sep="\t", dtype=str, chunksize=CHUNK_ROWS):
        # skip the rows before `processed` inside this chunk
        if processed > 0 and rows_seen + len(chunk) <= processed:
            rows_seen += len(chunk)
            continue
        if processed > 0 and rows_seen < processed:
            # keep only the remaining part
            chunk = chunk.iloc[(processed - rows_seen):].copy()
            rows_seen = processed

        # apply LIMIT
        if LIMIT_ROWS > 0:
            remain = LIMIT_ROWS - (processed + rows_added)
            if remain <= 0:
                break
            if len(chunk) > remain:
                chunk = chunk.iloc[:remain].copy()

        # the columns we need
        if "id" not in chunk.columns or "text" not in chunk.columns:
            raise ValueError(f"TSV must have columns id/text. got={list(chunk.columns)}")

        ids = chunk["id"].fillna("").tolist()
        texts = chunk["text"].fillna("").tolist()
        titles = chunk["title"].fillna("").tolist() if "title" in chunk.columns else [""] * len(texts)

        # batch processing
        meta_records_buffer = []
        for start in range(0, len(texts), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(texts))

            batch_inputs = []
            for i in range(start, end):
                t = titles[i].strip()
                x = texts[i].strip()
                if USE_TITLE and t:
                    batch_inputs.append(f"{t} [SEP] {x}")
                else:
                    batch_inputs.append(x)

            emb = encode_batch(batch_inputs, tokenizer, model, device, MAX_LENGTH, use_fp16)
            index.add(emb)

            # append meta in the same order
            for i in range(start, end):
                meta_records_buffer.append({
                    "pid": ids[i],
                    "title": titles[i],
                    "text": texts[i],
                })

            rows_added += (end - start)
            pbar.update(end - start)

            # flush periodically so the buffer does not grow too large
            if len(meta_records_buffer) >= 50_000:
                append_jsonl(meta_path, meta_records_buffer)
                meta_records_buffer = []

        # flush at the end of the chunk
        if meta_records_buffer:
            append_jsonl(meta_path, meta_records_buffer)

        # checkpoint the progress and the index
        processed_now = processed + rows_added
        write_processed_count(progress_path, processed_now)

        # checkpoint the index about once per chunk; more often would be slow
        faiss.write_index(index, index_path)

        rows_seen += len(chunk)

    pbar.close()

    t_end = time.time()
    secs = t_end - t_start
    total_rows = processed + rows_added
    rps = (rows_added / secs) if secs > 0 else 0.0

    print("\n✅ DONE")
    print(f"Index saved to: {index_path}")
    print(f"Meta  saved to: {meta_path}")
    print(f"Progress file : {progress_path}")
    print(f"Total rows now: {total_rows} (added {rows_added} this run)")
    print(f"Elapsed       : {secs/60:.1f} min")
    print(f"Throughput    : {rps:.1f} rows/sec")


if __name__ == "__main__":
    main()
