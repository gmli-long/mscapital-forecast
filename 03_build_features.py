"""Feature engineering: aggregate market/order/transaction time series per sample_id.

Uses Polars streaming group_by to stay within ~10GB RAM on 100M+ row files.
Output: parquet files with one row per sample_id.
"""
import polars as pl
import os
import time

from config import DATA_DIR, FEATURES_DIR

BASE = str(DATA_DIR)
OUT = str(FEATURES_DIR)
os.makedirs(OUT, exist_ok=True)

SBP = pl.col("seconds_before_predict")


def _last(x):
    """value at the bar closest to prediction time (min seconds_before_predict)."""
    return x.gather(SBP.arg_min()).first()


def _first(x):
    """value at the earliest bar (max seconds_before_predict)."""
    return x.gather(SBP.arg_max()).first()


def market_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    lf = lf.with_columns(
        pl.col("transaction_avgprice").cast(pl.Float64),
        ((pl.col("bid_price_1") + pl.col("ask_price_1")) / 2.0).alias("mid"),
        (pl.col("ask_price_1") - pl.col("bid_price_1")).alias("spread"),
        ((pl.col("bid_price_1") * pl.col("ask_volume_1") + pl.col("ask_price_1") * pl.col("bid_volume_1"))
         / (pl.col("ask_volume_1") + pl.col("bid_volume_1"))).alias("microprice"),
        ((pl.col("bid_volume_1") - pl.col("ask_volume_1"))
         / (pl.col("bid_volume_1") + pl.col("ask_volume_1"))).alias("imb"),
        (pl.col("bid_volume_1") + pl.col("ask_volume_1")).alias("depth"),
        (pl.col("transaction_count") > 0).cast(pl.Int32).alias("has_trade"),
        pl.col("transaction_volume").cast(pl.Int64),
    )
    w30 = (-SBP / 30.0).exp()
    w600 = (-SBP / 600.0).exp()

    return lf.group_by("sample_id").agg(
        pl.len().alias("m_bars"),
        SBP.max().alias("m_span"),
        # last-snapshot (closest to prediction time)
        _last(pl.col("mid")).alias("m_mid_last"),
        _first(pl.col("mid")).alias("m_mid_first"),
        (_last(pl.col("mid")) - _first(pl.col("mid"))).alias("m_mid_change"),
        _last(pl.col("microprice")).alias("m_micro_last"),
        _last(pl.col("spread")).alias("m_spread_last"),
        _last(pl.col("imb")).alias("m_imb_last"),
        _last(pl.col("depth")).alias("m_depth_last"),
        _last(pl.col("ask_price_1")).alias("m_ask1_last"),
        _last(pl.col("bid_price_1")).alias("m_bid1_last"),
        (_last(pl.col("ask_price_1")) - _last(pl.col("bid_price_1"))).alias("m_raw_spread_last"),
        # mid/stats
        pl.col("mid").mean().alias("m_mid_mean"),
        pl.col("mid").std().alias("m_mid_std"),
        pl.col("mid").min().alias("m_mid_min"),
        pl.col("mid").max().alias("m_mid_max"),
        pl.col("spread").mean().alias("m_spread_mean"),
        pl.col("spread").std().alias("m_spread_std"),
        pl.col("imb").mean().alias("m_imb_mean"),
        pl.col("imb").std().alias("m_imb_std"),
        pl.col("depth").mean().alias("m_depth_mean"),
        pl.col("depth").std().alias("m_depth_std"),
        pl.col("microprice").std().alias("m_micro_std"),
        # trade activity within bars
        pl.col("transaction_avgprice").drop_nulls().mean().alias("m_tavg_mean"),
        pl.col("transaction_avgprice").drop_nulls().std().alias("m_tavg_std"),
        pl.col("transaction_volume").sum().alias("m_tvol_sum"),
        pl.col("transaction_count").sum().alias("m_tcnt_sum"),
        pl.col("has_trade").sum().alias("m_trade_bars"),
        pl.col("has_trade").mean().alias("m_trade_bars_frac"),
        # time-weighted (recent emphasis)
        (pl.col("mid") * w30).sum() / w30.sum().alias("m_mid_w30"),
        (pl.col("imb") * w30).sum() / w30.sum().alias("m_imb_w30"),
        (pl.col("spread") * w30).sum() / w30.sum().alias("m_spread_w30"),
        (pl.col("mid") * w600).sum() / w600.sum().alias("m_mid_w600"),
        ((pl.col("mid") * w30).sum() / w30.sum() - (pl.col("mid") * w600).sum() / w600.sum()).alias("m_mid_slope"),
    )


def order_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    lf = lf.with_columns(
        pl.col("volume").cast(pl.Int64),
        pl.when(pl.col("side") == 0).then(pl.col("volume")).otherwise(-pl.col("volume")).alias("signed_vol"),
        pl.when(pl.col("order_action") == 0).then(pl.col("volume")).otherwise(-pl.col("volume")).alias("action_vol"),
        (pl.col("volume") * pl.col("price")).alias("vol_price"),
    )
    w = (-SBP / 30.0).exp()
    return lf.group_by("sample_id").agg(
        pl.len().alias("o_cnt"),
        pl.col("volume").sum().alias("o_vol"),
        pl.col("vol_price").sum().alias("o_vol_price"),
        (pl.col("side") == 0).sum().alias("o_buy_cnt"),
        (pl.col("side") == 1).sum().alias("o_sell_cnt"),
        pl.col("signed_vol").sum().alias("o_net_vol"),
        pl.col("signed_vol").abs().sum().alias("o_abs_vol"),
        (pl.col("signed_vol").sum() / pl.col("volume").sum()).alias("o_ofi_ratio"),
        (pl.col("order_action") == 0).sum().alias("o_new_cnt"),
        (pl.col("order_action") == 1).sum().alias("o_cancel_cnt"),
        pl.col("action_vol").sum().alias("o_net_action_vol"),
        _last(pl.col("price")).alias("o_price_last"),
        pl.col("price").mean().alias("o_price_mean"),
        pl.col("price").std().alias("o_price_std"),
        pl.col("price").min().alias("o_price_min"),
        pl.col("price").max().alias("o_price_max"),
        (pl.col("vol_price").sum() / pl.col("volume").sum()).alias("o_vwap"),
        pl.col("signed_vol").filter(SBP < 10).sum().alias("o_net_vol_10s"),
        pl.col("volume").filter(SBP < 10).sum().alias("o_vol_10s"),
        (pl.col("signed_vol") * w).sum().alias("o_netvol_w30"),
        (pl.col("volume") * w).sum().alias("o_vol_w30"),
    )


def transaction_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    lf = lf.with_columns(
        pl.col("volume").cast(pl.Int64),
        pl.when(pl.col("side") == 0).then(pl.col("volume")).otherwise(-pl.col("volume")).alias("signed_vol"),
        (pl.col("volume") * pl.col("price")).alias("vol_price"),
    )
    w = (-SBP / 30.0).exp()
    return lf.group_by("sample_id").agg(
        pl.len().alias("t_cnt"),
        pl.col("volume").sum().alias("t_vol"),
        pl.col("vol_price").sum().alias("t_vol_price"),
        (pl.col("side") == 0).sum().alias("t_buy_cnt"),
        (pl.col("side") == 1).sum().alias("t_sell_cnt"),
        (pl.col("side") == 0).sum() / pl.len().alias("t_buy_frac"),
        pl.col("signed_vol").sum().alias("t_net_vol"),
        pl.col("signed_vol").abs().sum().alias("t_abs_vol"),
        (pl.col("signed_vol").sum() / pl.col("volume").sum()).alias("t_imbalance"),
        _last(pl.col("price")).alias("t_price_last"),
        pl.col("price").mean().alias("t_price_mean"),
        pl.col("price").std().alias("t_price_std"),
        pl.col("price").min().alias("t_price_min"),
        pl.col("price").max().alias("t_price_max"),
        (pl.col("vol_price").sum() / pl.col("volume").sum()).alias("t_vwap"),
        pl.col("signed_vol").filter(SBP < 10).sum().alias("t_net_vol_10s"),
        pl.col("volume").filter(SBP < 10).sum().alias("t_vol_10s"),
        (pl.col("signed_vol") * w).sum().alias("t_netvol_w30"),
        (pl.col("volume") * w).sum().alias("t_vol_w30"),
    )


BUILDERS = {
    "market": market_features,
    "order": order_features,
    "transaction": transaction_features,
}


def build(split: str, name: str, head_n: int | None = None):
    path = BASE + f"\\{split}\\{name}.feather"
    lf = pl.scan_ipc(path)
    if head_n is not None:
        lf = lf.head(head_n)
    out = BUILDERS[name](lf).collect(engine="streaming").sort("sample_id")
    if head_n is None:
        outp = os.path.join(OUT, f"{split}_{name}.parquet")
        out.write_parquet(outp)
        print(f"[saved] {outp}  shape={out.shape}")
    else:
        print(f"[test head] {name}  shape={out.shape}")
        print(out.head(3))


if __name__ == "__main__":
    print("### VALIDATION (head slices) ###")
    for name in ["market", "order", "transaction"]:
        build("train", name, head_n=200_000)

    if os.environ.get("FULL") == "1":
        print("\n### FULL BUILD ###")
        for split in ["train", "test"]:
            for name in ["market", "order", "transaction"]:
                t = time.time()
                build(split, name)
                print(f"  {split}/{name}  {time.time()-t:.1f}s")