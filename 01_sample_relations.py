import pyarrow.feather as feather
import pandas as pd
import numpy as np

BASE = r"C:\Users\21597\.cache\kagglehub\competitions\ms-capital-real-financial-market-forecasting"

sub = pd.read_csv(BASE + "\\submission.csv")
print("=== submission.csv ===")
print("shape:", sub.shape)
print("columns:", list(sub.columns))
print(sub.head(5))
print("sample_id min/max/unique:", sub["sample_id"].min(), sub["sample_id"].max(), sub["sample_id"].nunique())

label = pd.read_feather(BASE + "\\train\\label.feather")
print("\n=== label: sample_id per month ===")
g = label.groupby("month")["sample_id"].agg(["min", "max", "count"])
print(g.head(8))
print("...")
print(g.tail(8))
print("\nper-month count describe:")
print(g["count"].describe())

print("\n=== train market/order/transaction sample_id range (narrow col) ===")
for name in ["market", "order", "transaction"]:
    col = feather.read_table(BASE + f"\\train\\{name}.feather", columns=["sample_id"])["sample_id"]
    mn = col.to_pandas().min()
    mx = col.to_pandas().max()
    nun = np.unique(col.to_pandas()).size
    print(f"{name:14s} rows={len(col):,}  sample_id min={mn} max={mx} unique={nun:,}")

print("\n=== test market/order/transaction sample_id range (narrow col) ===")
for name in ["market", "order", "transaction"]:
    col = feather.read_table(BASE + f"\\test\\{name}.feather", columns=["sample_id"])["sample_id"]
    mn = col.to_pandas().min()
    mx = col.to_pandas().max()
    nun = np.unique(col.to_pandas()).size
    print(f"{name:14s} rows={len(col):,}  sample_id min={mn} max={mx} unique={nun:,}")