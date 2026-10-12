import polars as pl
import numpy as np

rng = np.random.default_rng(0)
n = 40
df = pl.DataFrame({
    'sample_id': np.repeat(np.arange(10), 4).astype(np.int32),
    'seconds_before_predict': (np.tile([600.0, 400.0, 200.0, 1.0], 10)).astype(np.float32),
    'transaction_avgprice': (rng.random(n) * 0.1 + 0.95).astype(np.float32),
    'transaction_volume': rng.integers(0, 1000, n).astype(np.int32),
    'transaction_count': rng.integers(0, 5, n).astype(np.int32),
    'ask_price_1': (rng.random(n) * 0.01 + 1.0).astype(np.float32),
    'ask_volume_1': rng.integers(0, 1000, n).astype(np.int32),
    'bid_price_1': (rng.random(n) * 0.01 + 0.99).astype(np.float32),
    'bid_volume_1': rng.integers(0, 1000, n).astype(np.int32),
})

SBP = pl.col("seconds_before_predict")

def _last(x):
    return x.gather(SBP.arg_min()).first()

def _first(x):
    return x.gather(SBP.arg_max()).first()

w30 = (-SBP / 30.0).exp()
w600 = (-SBP / 600.0).exp()
mid = pl.col("mid")

AGG = [
    ("len", pl.len().alias("m_bars")),
    ("span", SBP.max().alias("m_span")),
    ("mid_last", _last(mid).alias("m_mid_last")),
    ("mid_first", _first(mid).alias("m_mid_first")),
    ("mid_change", (_last(mid) - _first(mid)).alias("m_mid_change")),
    ("micro_last", _last(pl.col("microprice")).alias("m_micro_last")),
    ("spread_last", _last(pl.col("spread")).alias("m_spread_last")),
    ("imb_last", _last(pl.col("imb")).alias("m_imb_last")),
    ("depth_last", _last(pl.col("depth")).alias("m_depth_last")),
    ("ask1_last", _last(pl.col("ask_price_1")).alias("m_ask1_last")),
    ("bid1_last", _last(pl.col("bid_price_1")).alias("m_bid1_last")),
    ("raw_spread", (_last(pl.col("ask_price_1")) - _last(pl.col("bid_price_1"))).alias("m_raw_spread_last")),
    ("mid_mean", mid.mean().alias("m_mid_mean")),
    ("mid_std", mid.std().alias("m_mid_std")),
    ("mid_min", mid.min().alias("m_mid_min")),
    ("mid_max", mid.max().alias("m_mid_max")),
    ("w30", (mid * w30).sum() / w30.sum().alias("m_mid_w30")),
    ("w600", (mid * w600).sum() / w600.sum().alias("m_mid_w600")),
    ("slope", ((mid * w30).sum() / w30.sum() - (mid * w600).sum() / w600.sum()).alias("m_mid_slope")),
]

lf = df.lazy().with_columns(
    pl.col("transaction_avgprice").cast(pl.Float64),
    ((pl.col("bid_price_1") + pl.col("ask_price_1")) / 2.0).alias("mid"),
    (pl.col("ask_price_1") - pl.col("bid_price_1")).alias("spread"),
    ((pl.col("bid_price_1") * pl.col("ask_volume_1") + pl.col("ask_price_1") * pl.col("bid_volume_1"))
     / (pl.col("ask_volume_1") + pl.col("bid_volume_1"))).alias("microprice"),
    ((pl.col("bid_volume_1") - pl.col("ask_volume_1"))
     / (pl.col("bid_volume_1") + pl.col("ask_volume_1"))).alias("imb"),
    (pl.col("bid_volume_1") + pl.col("ask_volume_1")).alias("depth"),
)

for i in range(len(AGG)):
    exprs = [e for _, e in AGG[:i + 1]]
    try:
        lf.group_by("sample_id").agg(exprs).collect(engine="streaming")
    except Exception as e:
        print(f"FAILED after adding [{i}] {AGG[i][0]}  -> {type(e).__name__}: {str(e)[:120]}")
        break
else:
    print("ALL OK")