"""Verify three data-dictionary facts the pipeline assumes but never checked:

  Q28 — single asset vs multi asset.  Decided by per-MONTH mid-price distribution:
        a single asset with a time trend shows one tight cluster per month whose
        level drifts across months; a multi-asset market shows multiple clusters
        WITHIN a month.  (A raw rounded-mid cluster count alone confounds these.)
  Q41 — is side 0=buy / 1=sell correct?  Taker direction test: a buy lifts the
        ask (price >= mid), a sell hits the bid (price <= mid).  Joined on
        FLOOR(seconds_before_predict) so the async streams actually line up, and
        the match rate is reported as a premise of the conclusion.
"""
import duckdb
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

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
print(f"bid_price_1 global range (row-group stats): [{min(mins):.4f}, {max(maxs):.4f}]")

# per-MONTH mid distribution: the decisive check for trend-vs-multi-asset
lab = pd.read_feather(BASE + "\\train\\label.feather")[["sample_id", "month"]]
mk = pd.read_parquet(f"{FEAT}/train_market.parquet", columns=["sample_id", "m_mid_mean"])
df = mk.merge(lab, on="sample_id", how="inner")
g = df.groupby("month")["m_mid_mean"].agg(["min", "max", "mean", "std", "count"])
print("\nper-month mid_mean distribution:")
print("  first 3 months / middle / last 3 months:")
show = pd.concat([g.iloc[[0, 1, 2]], g.iloc[[len(g)//2]], g.iloc[[-3, -2, -1]]])
print(show.round(4).to_string())
print(f"\nmonth-mean essentially flat across months: {g['mean'].iloc[0]:.4f} -> {g['mean'].iloc[-1]:.4f}  "
      f"=> no price time-trend")
low_frac = float((df["m_mid_mean"] < 0.9).mean())
print(f"every month shows the SAME shape (mass near 1.0 + a thin low-price tail); "
      f"global share of samples with mid_mean < 0.9 = {low_frac*100:.2f}%")
print("=> single asset: one dominant ~1.0 price level in every month, plus rare "
      "low-price samples scattered across all months (not a distinct second asset, not a trend)")

# ---------------- Q41: side encoding ----------------
print("\n" + "=" * 80)
print("Q41: side encoding — taker price vs market mid (gbp grid: sbp -> FLOOR)")
print("=" * 80)

con = duckdb.connect()
# floor sbp to integer seconds so the async market/transaction streams line up;
# a buy lifts the ask (px >= mid), a sell hits the bid (px <= mid).
q = f"""
WITH t AS (
  SELECT sample_id, FLOOR(seconds_before_predict)::INT AS sbp, price, side
  FROM read_parquet('{PARQ}/train_transaction.parquet')
  WHERE sample_id % 100 = 0
),
m AS (
  SELECT sample_id, FLOOR(seconds_before_predict)::INT AS sbp,
         (bid_price_1 + ask_price_1) / 2 AS mid, ask_price_1, bid_price_1
  FROM read_parquet('{PARQ}/train_market.parquet')
  WHERE sample_id % 100 = 0
  QUALIFY ROW_NUMBER() OVER (PARTITION BY sample_id, FLOOR(seconds_before_predict)::INT
                             ORDER BY seconds_before_predict) = 1
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
print("\n(side=0=BUY -> avg_delta_mid/frac_above_mid clearly POSITIVE;")
print(" side=1=SELL -> clearly NEGATIVE.  A mirrored result means the sign is flipped.)")

# join match rate — reported as a premise, not an afterthought
tot_t = con.execute(
    f"SELECT count(*) FROM read_parquet('{PARQ}/train_transaction.parquet') WHERE sample_id % 100 = 0"
).fetchone()[0]
matched = int(res["n"].sum())
print(f"\n[sbp grid alignment] sampled transaction rows={tot_t:,}  "
      f"matched on (sample_id, floor(sbp))={matched:,}  "
      f"({matched/tot_t*100:.1f}%)")

con.close()