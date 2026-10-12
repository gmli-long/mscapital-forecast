import pyarrow.ipc as ipc
import pyarrow.feather as feather
import pandas as pd
import numpy as np

from config import DATA_DIR

BASE = str(DATA_DIR)

FILES = [
    "train/label",
    "train/market",
    "train/order",
    "train/transaction",
    "test/market",
    "test/order",
    "test/transaction",
]

print("=" * 100)
print("SCHEMA + FILE METADATA")
print("=" * 100)
for name in FILES:
    path = BASE + "\\" + name + ".feather"
    with open(path, "rb") as f:
        reader = ipc.open_file(f)
        schema = reader.schema
        print(f"\n### {name}  |  num_record_batches={reader.num_record_batches}")
        for field in schema:
            print(f"    {field.name:24s} {field.type}")

print("\n" + "=" * 100)
print("ROW COUNTS (single narrow column read)")
print("=" * 100)
for name in FILES:
    path = BASE + "\\" + name + ".feather"
    n = feather.read_table(path, columns=["sample_id"]).num_rows
    print(f"{name:24s} rows = {n:,}")

print("\n" + "=" * 100)
print("LABEL SAMPLE (first rows + describe)")
print("=" * 100)
label = pd.read_feather(BASE + "\\train\\label.feather")
print(label.head(15))
print(label.dtypes)
print(label.describe(include="all"))
print("\nlabel.shape =", label.shape)
print("unique months:", np.sort(label["month"].unique()))
print("n samples:", label["sample_id"].nunique(), " sample_id dtype:", label["sample_id"].dtype)