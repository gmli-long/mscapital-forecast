"""Multi-fold expanding walk-forward temporal CV to assess generalization.

6 folds, each validates the NEXT 7 months against all prior months:
  fold 0: train 0-28   val 29-35
  fold 1: train 0-35   val 36-42
  fold 2: train 0-42   val 43-49
  fold 3: train 0-49   val 50-56
  fold 4: train 0-56   val 57-63
  fold 5: train 0-63   val 64-70   (== the single-fold baseline)
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import time
import os

from config import DATA_DIR, FEATURES_DIR, REPO_DIR

BASE = str(DATA_DIR)
FEAT = str(FEATURES_DIR)
OUT = str(REPO_DIR / "cv_results.csv")

label = pd.read_feather(BASE + "\\train\\label.feather")

feats = None
for name in ["market", "order", "transaction"]:
    f = pd.read_parquet(os.path.join(FEAT, f"train_{name}.parquet")).set_index("sample_id")
    feats = f if feats is None else feats.join(f, how="left")

X = feats.reset_index().merge(label[["sample_id", "month", "target"]], on="sample_id", how="inner")
feature_cols = [c for c in X.columns if c not in ("sample_id", "month", "target")]
print("merged:", X.shape, " n_features:", len(feature_cols))

params = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.05,
    num_leaves=127,
    min_data_in_leaf=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbosity=-1,
    seed=42,
)

# folds: (train_month_hi, val_months)   train = months <= train_month_hi
folds = [
    (28, [29, 30, 31, 32, 33, 34, 35]),
    (35, [36, 37, 38, 39, 40, 41, 42]),
    (42, [43, 44, 45, 46, 47, 48, 49]),
    (49, [50, 51, 52, 53, 54, 55, 56]),
    (56, [57, 58, 59, 60, 61, 62, 63]),
    (63, [64, 65, 66, 67, 68, 69, 70]),
]

rows = []
for i, (tr_hi, val_ms) in enumerate(folds):
    tr = X[X["month"] <= tr_hi]
    va = X[X["month"].isin(val_ms)]
    dtr = lgb.Dataset(tr[feature_cols], tr["target"])
    dva = lgb.Dataset(va[feature_cols], va["target"], reference=dtr)

    t = time.time()
    model = lgb.train(
        params, dtr, num_boost_round=2000, valid_sets=[dva],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    pred = model.predict(va[feature_cols], num_iteration=model.best_iteration)
    y = va["target"].to_numpy()

    rmse = np.sqrt(np.mean((pred - y) ** 2))
    mae = np.mean(np.abs(pred - y))
    pear = np.corrcoef(pred, y)[0, 1]
    spr = pd.Series(pred).corr(pd.Series(y), method="spearman")
    r2 = 1 - np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2)
    base0 = np.sqrt(np.mean(y ** 2))

    rows.append(dict(
        fold=i, train_months=f"0-{tr_hi}", val_months=f"{val_ms[0]}-{val_ms[-1]}",
        n_train=len(tr), n_val=len(va), best_iter=model.best_iteration,
        rmse=rmse, mae=mae, pearson=pear, spearman=spr, r2=r2, base0_rmse=base0,
        secs=round(time.time() - t, 1),
    ))
    print(f"fold {i}: train 0-{tr_hi} ({len(tr):,}) -> val {val_ms[0]}-{val_ms[-1]} "
          f"({len(va):,})  iter={model.best_iteration:4d}  "
          f"RMSE={rmse:.6f}  MAE={mae:.6f}  r={pear:.4f}  R2={r2:.4f}  ({rows[-1]['secs']}s)")

res = pd.DataFrame(rows)
res.to_csv(OUT, index=False)
print("\n[saved]", OUT)
print("\n=== summary across folds ===")
for c in ["rmse", "mae", "pearson", "spearman", "r2", "best_iter"]:
    print(f"{c:12s} mean={res[c].mean():.6f}  std={res[c].std():.6f}  min={res[c].min():.6f}  max={res[c].max():.6f}")

print("\nmean base0 RMSE:", res["base0_rmse"].mean())
print("model vs base0 RMSE ratio:", (res["rmse"] / res["base0_rmse"]).mean())