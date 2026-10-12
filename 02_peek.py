import polars as pl

BASE = r"C:\Users\21597\.cache\kagglehub\competitions\ms-capital-real-financial-market-forecasting"

for split in ["train", "test"]:
    for name in ["market", "order", "transaction"]:
        path = BASE + f"\\{split}\\{name}.feather"
        lf = pl.scan_ipc(path)
        h = lf.head(6).collect(engine="streaming")
        print(f"\n===== {split}/{name} : first 6 rows =====")
        print(h)
        # rows per sample and seconds range on a small head slice
        seg = lf.head(200000).collect(engine="streaming")
        cnt = seg.group_by("sample_id").agg(pl.len().alias("n"))
        sbp = seg.group_by("sample_id").agg(
            pl.col("seconds_before_predict").min().alias("sbp_min"),
            pl.col("seconds_before_predict").max().alias("sbp_max"),
        )
        print(f"[head-200k slice] distinct sample_ids={cnt.height}, rows/sample min/med/max:")
        print(cnt["n"].describe())
        print("seconds_before_predict per-sample min/max (first few):")
        print(sbp.head(6))
        # are seconds_before_predict integer-grid or float?
        print("seconds_before_predict unique sample of values:", seg["seconds_before_predict"].unique().sort().head(20).to_list())