"""Top-K feature pruning scan: find the best subset size.

Sorts features by cross-fold mean gain-share (from feature_stability.csv),
retrains the same 6-fold fixed-155 walk-forward CV for K in a grid, and reports
RMSE / Pearson / Spearman to locate the sweet spot.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import time
import os

from config import DATA_DIR, FEATURES_DIR, REPO_DIR

BASE = str(DATA_DIR)
FEAT = str(FEATURES_DIR)
STAB = str(REPO_DIR / "feature_stability.csv")
OUT = str(REPO_DIR / "prune_scan.csv")

NUM_ROUNDS = 155
K_GRID = [10, 15, 20, 25, 30, 40, 50, 61, 71]

label = pd.read_feather(BASE + "\\train\\label.feather")
feats = None
for name in ["market", "order", "transaction"]:
    f = pd.read_parquet(os.path.join(FEAT, f"train_{name}.parquet")).set_index("sample_id")
    feats = f if feats is None else feats.join(f, how="left")

X = feats.reset_index().merge(label[["sample_id", "month", "target"]], on="sample_id", how="inner")
feature_cols = [c for c in X.columns if c not in ("sample_id", "month", "target")]

# rank features by cross-fold stable importance
stab = pd.read_csv(STAB).sort_values("mean_share", ascending=False).reset_index(drop=True)
ranked = stab["feature"].tolist()

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

folds = [
    (28, [29, 30, 31, 32, 33, 34, 35]),
    (35, [36, 37, 38, 39, 40, 41, 42]),
    (42, [43, 44, 45, 46, 47, 48, 49]),
    (49, [50, 51, 52, 53, 54, 55, 56]),
    (56, [57, 58, 59, 60, 61, 62, 63]),
    (63, [64, 65, 66, 67, 68, 69, 70]),
]


def cv_run(cols):
    rmses, pears, sprs = [], [], []
    for tr_hi, val_ms in folds:
        tr = X[X["month"] <= tr_hi]
        va = X[X["month"].isin(val_ms)]
        dtr = lgb.Dataset(tr[cols], tr["target"])
        model = lgb.train(params, dtr, num_boost_round=NUM_ROUNDS)
        pred = model.predict(va[cols], num_iteration=NUM_ROUNDS)
        y = va["target"].to_numpy()
        rmses.append(float(np.sqrt(np.mean((pred - y) ** 2))))
        pears.append(float(np.corrcoef(pred, y)[0, 1]))
        sprs.append(float(pd.Series(pred).corr(pd.Series(y), method="spearman")))
    return np.mean(rmses), np.std(rmses), np.mean(pears), np.mean(sprs)


t0 = time.time()
rows = []
for K in K_GRID:
    cols = ranked[:K]
    rmse_m, rmse_s, pear, spr = cv_run(cols)
    rows.append(dict(K=K, n_features=len(cols), rmse=rmse_m, rmse_std=rmse_s,
                     pearson=pear, spearman=spr))
    print(f"K={K:3d}  n={len(cols):2d}  RMSE={rmse_m:.6f}±{rmse_s:.6f}  "
          f"Pearson={pear:.5f}  Spearman={spr:.5f}")

res = pd.DataFrame(rows)
res.to_csv(OUT, index=False)
print("\n[saved]", OUT)

full = res[res["n_features"] == len(feature_cols)].iloc[0]
best_rmse = res.loc[res["rmse"].idxmin()]
best_pear = res.loc[res["pearson"].idxmax()]
print(f"\nfull ({int(full['K'])} feats): RMSE={full['rmse']:.6f}  Pearson={full['pearson']:.5f}")
print(f"best RMSE : K={int(best_rmse['K'])}  RMSE={best_rmse['rmse']:.6f}  Pearson={best_rmse['pearson']:.5f}")
print(f"best Pearson: K={int(best_pear['K'])}  RMSE={best_pear['rmse']:.6f}  Pearson={best_pear['pearson']:.5f}")
print(f"\ntotal {time.time()-t0:.1f}s")