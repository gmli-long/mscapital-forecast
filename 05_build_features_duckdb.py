"""Build per-sample features via DuckDB SQL aggregation over parquet files."""
import duckdb
import os
import time

PARQ = r"c:\Users\21597\Desktop\spider_redbook\mscapital_forecast\parquet"
OUT = r"c:\Users\21597\Desktop\spider_redbook\mscapital_forecast\features"
os.makedirs(OUT, exist_ok=True)


def market_sql(path):
    return f"""
WITH t AS (
  SELECT sample_id, seconds_before_predict AS sbp,
    (bid_price_1 + ask_price_1)/2.0 AS mid,
    (ask_price_1 - bid_price_1) AS spread,
    (bid_price_1*ask_volume_1 + ask_price_1*bid_volume_1)/NULLIF((ask_volume_1+bid_volume_1),0) AS microprice,
    (bid_volume_1 - ask_volume_1)/NULLIF((bid_volume_1+ask_volume_1),0) AS imb,
    (bid_volume_1 + ask_volume_1) AS depth,
    transaction_avgprice, transaction_volume, transaction_count, ask_price_1, bid_price_1
  FROM read_parquet('{path}')
)
SELECT sample_id,
  count(*) AS m_bars,
  max(sbp) AS m_span,
  arg_min(mid, sbp) AS m_mid_last,
  arg_max(mid, sbp) AS m_mid_first,
  arg_min(mid, sbp) - arg_max(mid, sbp) AS m_mid_change,
  arg_min(microprice, sbp) AS m_micro_last,
  arg_min(spread, sbp) AS m_spread_last,
  arg_min(imb, sbp) AS m_imb_last,
  arg_min(depth, sbp) AS m_depth_last,
  arg_min(ask_price_1, sbp) AS m_ask1_last,
  arg_min(bid_price_1, sbp) AS m_bid1_last,
  avg(mid) AS m_mid_mean,
  stddev(mid) AS m_mid_std,
  min(mid) AS m_mid_min,
  max(mid) AS m_mid_max,
  avg(spread) AS m_spread_mean,
  stddev(spread) AS m_spread_std,
  avg(imb) AS m_imb_mean,
  stddev(imb) AS m_imb_std,
  avg(depth) AS m_depth_mean,
  stddev(depth) AS m_depth_std,
  stddev(microprice) AS m_micro_std,
  avg(transaction_avgprice) AS m_tavg_mean,
  stddev(transaction_avgprice) AS m_tavg_std,
  sum(transaction_volume) AS m_tvol_sum,
  sum(transaction_count) AS m_tcnt_sum,
  sum(CASE WHEN transaction_count>0 THEN 1 ELSE 0 END) AS m_trade_bars,
  avg(CASE WHEN transaction_count>0 THEN 1.0 ELSE 0.0 END) AS m_trade_bars_frac,
  sum(mid*exp(-sbp/30.0))/sum(exp(-sbp/30.0)) AS m_mid_w30,
  sum(imb*exp(-sbp/30.0))/sum(exp(-sbp/30.0)) AS m_imb_w30,
  sum(spread*exp(-sbp/30.0))/sum(exp(-sbp/30.0)) AS m_spread_w30,
  sum(mid*exp(-sbp/600.0))/sum(exp(-sbp/600.0)) AS m_mid_w600,
  sum(mid*exp(-sbp/30.0))/sum(exp(-sbp/30.0)) - sum(mid*exp(-sbp/600.0))/sum(exp(-sbp/600.0)) AS m_mid_slope
FROM t GROUP BY sample_id
"""


def order_sql(path):
    return f"""
WITH t AS (
  SELECT sample_id, seconds_before_predict AS sbp, price, volume, side, order_action
  FROM read_parquet('{path}')
)
SELECT sample_id,
  count(*) AS o_cnt,
  sum(volume) AS o_vol,
  sum(volume*price) AS o_vol_price,
  sum(CASE WHEN side=0 THEN 1 ELSE 0 END) AS o_buy_cnt,
  sum(CASE WHEN side=1 THEN 1 ELSE 0 END) AS o_sell_cnt,
  sum(CASE WHEN side=0 THEN volume ELSE -volume END) AS o_net_vol,
  sum(CASE WHEN side=0 THEN volume ELSE -volume END)/NULLIF(sum(volume),0) AS o_ofi_ratio,
  sum(CASE WHEN order_action=0 THEN 1 ELSE 0 END) AS o_new_cnt,
  sum(CASE WHEN order_action=1 THEN 1 ELSE 0 END) AS o_cancel_cnt,
  sum(CASE WHEN order_action=0 THEN volume ELSE -volume END) AS o_net_action_vol,
  arg_min(price, sbp) AS o_price_last,
  avg(price) AS o_price_mean,
  stddev(price) AS o_price_std,
  min(price) AS o_price_min,
  max(price) AS o_price_max,
  sum(volume*price)/NULLIF(sum(volume),0) AS o_vwap,
  sum(CASE WHEN sbp<10 THEN (CASE WHEN side=0 THEN volume ELSE -volume END) ELSE 0 END) AS o_net_vol_10s,
  sum(CASE WHEN sbp<10 THEN volume ELSE 0 END) AS o_vol_10s,
  sum((CASE WHEN side=0 THEN volume ELSE -volume END)*exp(-sbp/30.0)) AS o_netvol_w30,
  sum(volume*exp(-sbp/30.0)) AS o_vol_w30
FROM t GROUP BY sample_id
"""


def transaction_sql(path):
    return f"""
WITH t AS (
  SELECT sample_id, seconds_before_predict AS sbp, price, volume, side
  FROM read_parquet('{path}')
)
SELECT sample_id,
  count(*) AS t_cnt,
  sum(volume) AS t_vol,
  sum(volume*price) AS t_vol_price,
  sum(CASE WHEN side=0 THEN 1 ELSE 0 END) AS t_buy_cnt,
  sum(CASE WHEN side=1 THEN 1 ELSE 0 END) AS t_sell_cnt,
  avg(CASE WHEN side=0 THEN 1.0 ELSE 0.0 END) AS t_buy_frac,
  sum(CASE WHEN side=0 THEN volume ELSE -volume END) AS t_net_vol,
  sum(CASE WHEN side=0 THEN volume ELSE -volume END)/NULLIF(sum(volume),0) AS t_imbalance,
  arg_min(price, sbp) AS t_price_last,
  avg(price) AS t_price_mean,
  stddev(price) AS t_price_std,
  min(price) AS t_price_min,
  max(price) AS t_price_max,
  sum(volume*price)/NULLIF(sum(volume),0) AS t_vwap,
  sum(CASE WHEN sbp<10 THEN (CASE WHEN side=0 THEN volume ELSE -volume END) ELSE 0 END) AS t_net_vol_10s,
  sum(CASE WHEN sbp<10 THEN volume ELSE 0 END) AS t_vol_10s,
  sum((CASE WHEN side=0 THEN volume ELSE -volume END)*exp(-sbp/30.0)) AS t_netvol_w30,
  sum(volume*exp(-sbp/30.0)) AS t_vol_w30
FROM t GROUP BY sample_id
"""


SQL = {"market": market_sql, "order": order_sql, "transaction": transaction_sql}


def build(split, name):
    path = os.path.join(PARQ, f"{split}_{name}.parquet")
    q = SQL[name](path) + " ORDER BY sample_id"
    dst = os.path.join(OUT, f"{split}_{name}.parquet")
    dst = dst.replace("\\", "/")
    duckdb.sql(f"COPY ({q}) TO '{dst}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    n = duckdb.sql(f"SELECT count(*) FROM read_parquet('{dst}')").fetchone()[0]
    print(f"[ok] {split}/{name}  rows={n:,}")


if __name__ == "__main__":
    for split in ["train", "test"]:
        for name in ["market", "order", "transaction"]:
            t = time.time()
            build(split, name)
            print(f"    {time.time()-t:.1f}s")