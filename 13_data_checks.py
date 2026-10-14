"""Verify two data-dictionary facts that the pipeline assumes but never checked:

  Q28 — is the market single-asset or multi-asset?  (read bid/ask price range
        from parquet row-group stats + per-sample mid from the features table)
  Q41 — is side 0=buy / 1=sell correct?  (transaction taker-price vs market mid
        sign test, plus net-flow vs target correlation as an economic sanity check)
"""
import duckdb
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow.feather as feather

from config import PARQUET_DIR, FEATURES_DIR, DATA_DIR

PARQ = str(PARQUET_DIR).replace("\\", "/")
FEAT = str(FEATURES_DIR)
BASE = str(DATA_DIR)

# ---------------- Q28: single vs multi asset ----------------
print("=" * 80)
print("Q28: price level — single asset vs multi asset")
print("=" * 80)

pf = pq.ParquetFile(f"{PARQ}/train_market.parquet")
mins, maxs = [], []
for i in range(pf.metadata.num_row_groups):
    rg = pf.metadata.row_group(i)
    for j in range(rg.num_columns):
        if rg.column(j).path_in_schema == "bid_price_1":
            s = rg.column(j).statistics
            if s and s.has_min_max:
                mins.append(float(s.min))
                maxs.append(float(s.max))
print(f"bid_price_1 global range (from row-group stats): [{min(mins):.4f}, {max(maxs):.4f}]")

mk = pd.read_parquet(f"{FEAT}/train_market.parquet",
                     columns=["m_mid_min", "m_mid_max", "m_mid_mean", "m_mid_std"])
print("\nper-sample mid statistics (train_market features):")
for c in ["m_mid_mean", "m_mid_std", "m_mid_min", "m_mid_max"]:
    d = mk[c]
    print(f"  {c:12s} min={d.min():.4f}  max={d.max():.4f}  mean={d.mean():.4f}  std={d.std():.6f}")

# distinct price clusters (rounded mid): single asset -> one tight cluster
clusters = mk["m_mid_mean"].round(1).value_counts().sort_index()
print(f"\ndistinct rounded-mid clusters: {len(clusters)}")
print(clusters.head(10).to_string())

# ---------------- Q41: side encoding ----------------
print("\n" + "=" * 80)
print("Q41: side encoding — taker price vs market mid sign test")
print("=" * 80)

con = duckdb.connect()
# transaction taker: a buy hits the ask (px >= mid), a sell hits the bid (px <= mid)
q = f"""
WITH t AS (
  SELECT sample_id, seconds_before_predict AS sbp, price, side FROM read_parquet('{PARQ}/train_transaction.parquet')
  WHERE sample_id % 100 = 0
),
m AS (
  SELECT sample_id, seconds_before_predict AS sbp, (bid_price_1 + ask_price_1) / 2 AS mid,
         ask_price_1, bid_price_1
  FROM read_parquet('{PARQ}/train_market.parquet')
  WHERE sample_id % 100 = 0
)
SELECT t.side,
       count(*) AS n,
       avg(t.price)                         AS avg_px,
       avg(t.price - m.mid)                 AS avg_delta_mid,
       avg(CASE WHEN t.price >  m.mid THEN 1.0 ELSE 0.0 END) AS frac_above_mid,
       avg(CASE WHEN t.price >= m.ask_price_1 THEN 1.0 ELSE 0.0 END) AS frac_at_or_above_ask
FROM t JOIN m ON t.sample_id = m.sample_id AND t.sbp = m.sbp
GROUP BY t.side ORDER BY t.side
"""
res = con.execute(q).df()
print(res.to_string(index=False))
print("\n(if side=0 is BUY -> its avg_delta_mid/frac_above_mid should be clearly POSITIVE;")
print(" if side=1 is SELL -> negative.  A mirrored result means the sign is flipped.)")

# join match rate (are the sbp grids aligned?)
join_n = con.execute(f"""
WITH t AS (SELECT sample_id, seconds_before_predict AS sbp FROM read_parquet('{PARQ}/train_transaction.parquet') WHERE sample_id % 100 = 0),
m AS (SELECT sample_id, seconds_before_predict AS sbp FROM read_parquet('{PARQ}/train_market.parquet') WHERE sample_id % 100 = 0)
SELECT count(*) FROM t JOIN m ON t.sample_id=m.sample_id AND t.sbp=m.sbp
""").fetchone()[0]
tot_t = con.execute(f"SELECT count(*) FROM read_parquet('{PARQ}/train_transaction.parquet') WHERE sample_id % 100 = 0").fetchone()[0]
print(f"\n[sbp grid alignment] sampled transaction rows={tot_t:,}  exact (sample_id,sbp) match with market={join_n:,}")

# economic sanity: does net-buy flow correlate with future return as expected?
print("\n--- economic sanity: net flow (side0-side1) vs target ---")
lab = pd.read_feather(BASE + "\\train\\label.feather")[["sample_id", "target"]]
for name, col in [("transaction", "t_net_vol"), ("order", "o_net_vol")]:
    f = pd.read_parquet(f"{FEAT}/train_{name}.parquet", columns=["sample_id", col])
    df = lab.merge(f, on="sample_id", how="inner")
    r = np.corrcoef(df[col], df["target"])[0, 1]
    print(f"  corr({col}, target) = {r:+.4f}   (positive -> side0=buy aligns with 'buy pressure -> up')")

con.close()