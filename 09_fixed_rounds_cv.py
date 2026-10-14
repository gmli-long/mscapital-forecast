"""Strict fixed-rounds walk-forward CV (no early stopping) to remove selection bias.

Same 6 folds as 08_multifold_cv.py, but every fold trains a FIXED number of
rounds (155 = fold 5's early-stopping best_iteration) with NO validation-based early stop,
so the validation fold is never used to select iterations. Compares against the
early-stopping version to quantify the optimism bias.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import time
import os

from config import REPO_DIR
from common import PARAMS, FOLDS, NUM_ROUNDS, load_X

OUT = str(REPO_DIR / "cv_fixed_results.csv")
CV_EARLY = str(REPO_DIR / "cv_results.csv")

X, feature_cols = load_X()

params = dict(PARAMS)
folds = FOLDS

rows = []
for i, (tr_hi, val_ms) in enumerate(folds):
    tr = X[X["month"] <= tr_hi]
    va = X[X["month"].isin(val_ms)]
    dtr = lgb.Dataset(tr[feature_cols], tr["target"])

    t = time.time()
    model = lgb.train(params, dtr, num_boost_round=NUM_ROUNDS)
    pred = model.predict(va[feature_cols], num_iteration=NUM_ROUNDS)
    y = va["target"].to_numpy()

    rmse = np.sqrt(np.mean((pred - y) ** 2))
    mae = np.mean(np.abs(pred - y))
    pear = np.corrcoef(pred, y)[0, 1]
    spr = pd.Series(pred).corr(pd.Series(y), method="spearman")
    r2 = 1 - np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2)
    base0 = np.sqrt(np.mean(y ** 2))

    rows.append(dict(
        fold=i, val_months=f"{val_ms[0]}-{val_ms[-1]}", n_train=len(tr), n_val=len(va),
        rounds=NUM_ROUNDS, rmse=rmse, mae=mae, pearson=pear, spearman=spr, r2=r2,
        base0_rmse=base0, secs=round(time.time() - t, 1),
    ))
    print(f"fold {i}: val {val_ms[0]}-{val_ms[-1]} ({len(va):,})  fixed={NUM_ROUNDS}  "
          f"RMSE={rmse:.6f}  r={pear:.4f}  R2={r2:.4f}  ({rows[-1]['secs']}s)")

res = pd.DataFrame(rows)
res.to_csv(OUT, index=False)
print("\n[saved]", OUT)

print("\n=== fixed-rounds summary ===")
for c in ["rmse", "mae", "pearson", "spearman", "r2"]:
    print(f"{c:12s} mean={res[c].mean():.6f}  std={res[c].std():.6f}")

# --- compare against early-stopping version, per fold ---
if os.path.exists(CV_EARLY):
    early = pd.read_csv(CV_EARLY)
    merged = res.merge(early, on="fold", suffixes=("_fixed", "_early"))
    print("\n=== bias: early-stop (optimistic) vs fixed rounds ===")
    print("per-fold RMSE diff (fixed - early), positive = early-stop looked better:")
    for _, r in merged.iterrows():
        d = r["rmse_fixed"] - r["rmse_early"]
        vm = r["val_months_fixed"] if "val_months_fixed" in merged else r["val_months_early"]
        print(f"  fold {int(r['fold'])}  val {vm}: fixed={r['rmse_fixed']:.6f} "
              f"early={r['rmse_early']:.6f}  diff={d:+.6f}")
    mean_fixed = merged["rmse_fixed"].mean()
    mean_early = merged["rmse_early"].mean()
    print(f"\nmean RMSE: fixed={mean_fixed:.6f}  early={mean_early:.6f}  "
          f"bias={mean_fixed - mean_early:+.6f} ({(mean_fixed/mean_early-1)*100:+.2f}%)")
    print(f"mean Pearson: fixed={merged['pearson_fixed'].mean():.5f}  "
          f"early={merged['pearson_early'].mean():.5f}")