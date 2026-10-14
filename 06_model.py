"""Load features, EDA, and train a LightGBM baseline with month-based temporal CV."""
import pandas as pd
import numpy as np
import lightgbm as lgb
import time

from config import REPO_DIR
from common import PARAMS, load_X

X, feature_cols = load_X()
print("merged:", X.shape, " n_features:", len(feature_cols))

# ---- EDA ----
print("\n=== target ===")
print(X["target"].describe())
print("\n=== top |corr| with target ===")
corr = X[feature_cols + ["target"]].corr(numeric_only=True)["target"].drop("target").abs().sort_values(ascending=False)
print(corr.head(25))
print("\n=== NaN fraction per feature (top 15) ===")
nanfrac = X[feature_cols].isna().mean().sort_values(ascending=False)
print(nanfrac.head(15))

# ---- temporal split ----
val_months = [64, 65, 66, 67, 68, 69, 70]
tr = X[~X["month"].isin(val_months)]
va = X[X["month"].isin(val_months)]
print(f"\ntrain months 0-63: {tr.shape},  val months 64-70: {va.shape}")

params = dict(PARAMS)

dtr = lgb.Dataset(tr[feature_cols], tr["target"])
dva = lgb.Dataset(va[feature_cols], va["target"], reference=dtr)

t = time.time()
model = lgb.train(
    params,
    dtr,
    num_boost_round=2000,
    valid_sets=[dva],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
)
print(f"trained in {time.time()-t:.1f}s")

pred = model.predict(va[feature_cols], num_iteration=model.best_iteration)
y = va["target"].to_numpy()

rmse = np.sqrt(np.mean((pred - y) ** 2))
mae = np.mean(np.abs(pred - y))
pear = np.corrcoef(pred, y)[0, 1]
spr = pd.Series(pred).corr(pd.Series(y), method="spearman")
r2 = 1 - np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2)
print("\n=== validation metrics (months 64-70) ===")
print(f"RMSE={rmse:.6f}  MAE={mae:.6f}  Pearson={pear:.5f}  Spearman={spr:.5f}  R2={r2:.5f}")

# baseline: predict 0 (mean)
print(f"\nbaseline predict-0: RMSE={np.sqrt(np.mean(y**2)):.6f}  MAE={np.mean(np.abs(y)):.6f}")

# feature importance
imp = pd.Series(model.feature_importance("gain"), index=feature_cols).sort_values(ascending=False)
print("\n=== top 30 importances ===")
print(imp.head(30).to_string())

model.save_model(str(REPO_DIR / "lgb_baseline.txt"))
print("\n[saved] lgb_baseline.txt")