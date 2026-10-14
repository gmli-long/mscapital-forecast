"""Statistical significance + seed sensitivity for the LightGBM model.

Answers the two questions that the raw numbers alone cannot:
  A. Is "better than the zero-prediction baseline" (RMSE 0.003230 vs 0.003258,
     ~0.9%) really a signal, or just noise?  -> Diebold-Mariano test + paired
     bootstrap on the squared-error differential, pooled over the 6-fold
     walk-forward at fixed 155 rounds.
  B. Is "beats the baseline" robust to the random seed, or a seed=42 accident?
     -> retrain the hardest split (train 0-63 / val 64-70) at 155 rounds across
     several seeds and check RMSE/base0 < 1 for every seed.

No scipy required: the two-sided normal p-value uses math.erf, and the bootstrap
uses numpy only.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import os
import time
from math import erf

from config import DATA_DIR, FEATURES_DIR, REPO_DIR

BASE = str(DATA_DIR)
FEAT = str(FEATURES_DIR)
NUM_ROUNDS = 155

label = pd.read_feather(BASE + "\\train\\label.feather")

feats = None
for name in ["market", "order", "transaction"]:
    f = pd.read_parquet(os.path.join(FEAT, f"train_{name}.parquet")).set_index("sample_id")
    feats = f if feats is None else feats.join(f, how="left")

X = feats.reset_index().merge(label[["sample_id", "month", "target"]], on="sample_id", how="inner")
feature_cols = [c for c in X.columns if c not in ("sample_id", "month", "target")]
print("merged:", X.shape, " n_features:", len(feature_cols))

params = dict(
    objective="regression", metric="rmse", learning_rate=0.05,
    num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    verbosity=-1, seed=42,
)

folds = [
    (28, [29, 30, 31, 32, 33, 34, 35]),
    (35, [36, 37, 38, 39, 40, 41, 42]),
    (42, [43, 44, 45, 46, 47, 48, 49]),
    (49, [50, 51, 52, 53, 54, 55, 56]),
    (56, [57, 58, 59, 60, 61, 62, 63]),
    (63, [64, 65, 66, 67, 68, 69, 70]),
]


def two_sided_p(z):
    return 1.0 - erf(abs(z) / np.sqrt(2.0))


def hac_se(d, max_lag=5):
    """Newey-West autocorrelation-robust std error of the mean of d."""
    n = len(d)
    d = d - d.mean()
    g0 = (d ** 2).sum() / n
    s = g0
    for lag in range(1, max_lag + 1):
        w = 1.0 - lag / (max_lag + 1.0)
        s += 2.0 * w * (d[lag:] * d[:-lag]).sum() / n
    return np.sqrt(max(s / n, 1e-30))


# ---------------- Part A: pooled DM + bootstrap over 6 folds ----------------
print("\n=== Part A: 6-fold walk-forward @ 155 (seed 42) ===")
y_all, pred_all = [], []
for i, (tr_hi, val_ms) in enumerate(folds):
    tr = X[X["month"] <= tr_hi]
    va = X[X["month"].isin(val_ms)]
    m = lgb.train(params, lgb.Dataset(tr[feature_cols], tr["target"]), num_boost_round=NUM_ROUNDS)
    p = m.predict(va[feature_cols], num_iteration=NUM_ROUNDS)
    y_all.append(va["target"].to_numpy())
    pred_all.append(p)
    rmse = float(np.sqrt(np.mean((p - va["target"].to_numpy()) ** 2)))
    print(f"  fold {i}  val {val_ms[0]}-{val_ms[-1]}  RMSE={rmse:.6f}")

y = np.concatenate(y_all)
pred = np.concatenate(pred_all)
n = len(y)

rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
base0 = float(np.sqrt(np.mean(y ** 2)))
d = y ** 2 - (pred - y) ** 2  # positive = model better than zero-pred baseline
dbar = float(d.mean())
se = hac_se(d)
z = dbar / se
p_dm = two_sided_p(z)

rel = 1.0 - rmse / base0
print(f"\npooled n={n:,}  RMSE={rmse:.6f}  baseline(zero-pred)={base0:.6f}  "
      f"improvement={rel*100:.2f}%")

print("\n--- Diebold-Mariano (HAC, lag<=5) ---")
print(f"mean loss diff d = {dbar:.8f}  SE = {se:.8f}  z = {z:.3f}  p_two_sided = {p_dm:.3g}")

print("\n--- paired bootstrap (B=5000) on mean loss diff ---")
rng = np.random.default_rng(42)
B = 5000
boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"bootstrap 95% CI of mean(d): [{lo:.8f}, {hi:.8f}]  (excludes 0 -> {'YES' if lo > 0 else 'NO'})")

print("\n--- Pearson correlation significance ---")
r = float(np.corrcoef(pred, y)[0, 1])
zr = np.arctanh(r)
se_r = 1.0 / np.sqrt(n - 3)
lo_r, hi_r = np.tanh(zr - 1.96 * se_r), np.tanh(zr + 1.96 * se_r)
p_r = two_sided_p(zr / se_r)
print(f"Pearson r = {r:.4f}  95% CI = [{lo_r:.4f}, {hi_r:.4f}]  z = {zr/se_r:.1f}  "
      f"p_two_sided = {p_r:.3g}")

# ---------------- Part B: seed sensitivity on fold 5 ----------------
print("\n=== Part B: seed sensitivity (train 0-63 -> val 64-70, @155) ===")
tr = X[X["month"] <= 63]
va = X[X["month"].isin([64, 65, 66, 67, 68, 69, 70])]
y5 = va["target"].to_numpy()
base0_5 = float(np.sqrt(np.mean(y5 ** 2)))
print(f"baseline(zero-pred) RMSE = {base0_5:.6f}")

rows = []
for sd in [42, 0, 1, 7, 123]:
    p = dict(params)
    p["seed"] = sd
    m = lgb.train(p, lgb.Dataset(tr[feature_cols], tr["target"]), num_boost_round=NUM_ROUNDS)
    pr = m.predict(va[feature_cols], num_iteration=NUM_ROUNDS)
    rm = float(np.sqrt(np.mean((pr - y5) ** 2)))
    pear = float(np.corrcoef(pr, y5)[0, 1])
    rows.append(dict(seed=sd, rmse=rm, ratio=rm / base0_5, pearson=pear))
    print(f"  seed={sd:3d}  RMSE={rm:.6f}  RMSE/base0={rm/base0_5:.4f}  Pearson={pear:.4f}")

res = pd.DataFrame(rows)
worst = res.loc[res["ratio"].idxmax()]
print(f"\nall seeds beat baseline: {'YES' if (res['ratio'] < 1).all() else 'NO'}")
print(f"ratio range: [{res['ratio'].min():.4f}, {res['ratio'].max():.4f}]  "
      f"worst seed={int(worst['seed'])}")

out = REPO_DIR / "significance_results.csv"
res.to_csv(str(out), index=False)
print("\n[saved]", out)