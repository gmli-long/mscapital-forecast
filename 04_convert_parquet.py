"""Convert feather -> parquet (row groups + zstd) for efficient DuckDB streaming aggregation."""
import pyarrow.feather as feather
import pyarrow.parquet as pq
import os
import time

from config import DATA_DIR, PARQUET_DIR

BASE = str(DATA_DIR)
OUT = str(PARQUET_DIR)

FILES = [
    ("train", "market"),
    ("train", "order"),
    ("train", "transaction"),
    ("test", "market"),
    ("test", "order"),
    ("test", "transaction"),
]

for split, name in FILES:
    src = BASE + f"\\{split}\\{name}.feather"
    dst = os.path.join(OUT, f"{split}_{name}.parquet")
    os.makedirs(OUT, exist_ok=True)
    t = time.time()
    tab = feather.read_table(src, memory_map=True)
    pq.write_table(tab, dst, row_group_size=1_000_000, compression="zstd")
    print(f"[ok] {split}/{name}  rows={tab.num_rows:,}  {time.time()-t:.1f}s  ({os.path.getsize(dst)/1e9:.2f} GB)")