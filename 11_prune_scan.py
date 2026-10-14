"""Top-K feature pruning scan: find the best subset size.

Sorts features by cross-fold mean gain-share (from feature_stability.csv),
retrains the same 6-fold fixed-155 walk-forward CV for K in a grid, and reports
RMSE / Pearson / Spearman to locate the sweet spot.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import time

from config import REPO_DIR
from common import PARAMS, FOLDS, NUM_ROUNDS, load_X

STAB = str(REPO_DIR / "feature_stability.csv")
OUT = str(REPO_DIR / "prune_scan.csv")

# 40 = 拐点/饱和点（K 再往上 RMSE 几乎不再降）；61 = 10_feature_stability 的
# cum_mean_share<=95% 规则选出的特征数。两者口径不同，都扫一遍以便对照。
K_GRID = [10, 15, 20, 25, 30, 40, 50, 61, 71]

X, feature_cols = load_X()

# rank features by cross-fold stable importance
stab = pd.read_csv(STAB).sort_values("mean_share", ascending=False).reset_index(drop=True)
ranked = stab["feature"].tolist()

params = dict(PARAMS)
folds = FOLDS


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