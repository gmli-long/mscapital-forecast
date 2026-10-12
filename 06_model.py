"""Load features, EDA, and train a LightGBM baseline with month-based temporal CV."""
import pandas as pd
import numpy as np
import lightgbm as lgb
import time
import os

from config import DATA_DIR, FEATURES_DIR, REPO_DIR

BASE = str(DATA_DIR)
FEAT = str(FEATURES_DIR)

label = pd.read_feather(BASE + "\\train\\label.feather")
print("label:", label.shape)

feats = None
for name in ["market", "order", "transaction"]:
    f = pd.read_parquet(os.path.join(FEAT, f"train_{name}.parquet"))
    print(f"train_{name}: {f.shape}")
    f = f.set_index("sample_id")
    if feats is None:
        feats = f
    else:
        feats = feats.join(f, how="left")

X = feats.reset_index().merge(label[["sample_id", "month", "target"]], on="sample_id", how="inner")
print("merged:", X.shape)

feature_cols = [c for c in X.columns if c not in ("sample_id", "month", "target")]
print("n_features:", len(feature_cols))

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