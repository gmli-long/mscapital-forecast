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

from config import REPO_DIR
from common import PARAMS, FOLDS, load_X

OUT = str(REPO_DIR / "cv_results.csv")

X, feature_cols = load_X()
print("merged:", X.shape, " n_features:", len(feature_cols))

params = dict(PARAMS)
folds = FOLDS

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