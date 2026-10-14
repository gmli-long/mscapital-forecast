"""Statistical significance + seed sensitivity for the LightGBM model.

Answers two questions the raw numbers alone cannot, and persists every number
to CSV artifacts (not just stdout) so the claims are reproducible:

  A. Is "better than the zero-prediction baseline" (~0.9%) a real signal?
     -> pooled 6-fold walk-forward loss-differential d = y^2 - (y-hat)^2,
        tested with Diebold-Mariano (Newey-West HAC, with a max-lag sensitivity
        sweep) and a moving-block bootstrap (time blocks, consistent with the
        HAC assumption that d is a time series).
  B. Is "beats the baseline" robust to the random seed?
     -> retrain the hardest split (train 0-63 / val 64-70) at 155 rounds over
        many seeds, single-threaded so the seed is the ONLY thing that changes,
        and check RMSE/base0 < 1 holds for every seed.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from math import erf

from config import REPO_DIR
from common import PARAMS, FOLDS, NUM_ROUNDS, load_X

X, feature_cols = load_X()
print("merged:", X.shape, " n_features:", len(feature_cols))

params = dict(PARAMS)
params["num_threads"] = 1  # single-threaded: seed is the only source of variation
folds = FOLDS


def two_sided_p(z):
    return 1.0 - erf(abs(z) / np.sqrt(2.0))


def hac_se(d, max_lag=5):
    """Newey-West HAC std error of the mean of a time-ordered d."""
    n = len(d)
    d = d - d.mean()
    g0 = (d ** 2).sum() / n
    s = g0
    for lag in range(1, max_lag + 1):
        w = 1.0 - lag / (max_lag + 1.0)
        s += 2.0 * w * (d[lag:] * d[:-lag]).sum() / n
    return np.sqrt(max(s / n, 1e-30))


def block_bootstrap(d, L, B, seed=42):
    """Moving-block bootstrap of the mean of d (keeps within-block autocorrelation)."""
    rng = np.random.default_rng(seed)
    n = len(d)
    n_blocks = n - L + 1
    cs = np.concatenate([[0.0], np.cumsum(d)])
    block_sums = cs[L:] - cs[:-L]
    nb = n // L
    means = np.empty(B)
    for b0 in range(0, B, 256):
        b1 = min(B, b0 + 256)
        starts = rng.integers(0, n_blocks, size=(b1 - b0, nb))
        means[b0:b1] = block_sums[starts].sum(axis=1) / (nb * L)
    return means


# ---------------- Part A: pooled DM + bootstrap over 6 folds ----------------
print(f"\n=== Part A: 6-fold walk-forward @ {NUM_ROUNDS} (seed 42) ===")
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
rel = 1.0 - rmse / base0
print(f"\npooled n={n:,}  RMSE={rmse:.6f}  baseline(zero-pred)={base0:.6f}  "
      f"improvement={rel*100:.2f}%")

# DM with a max-lag sweep: the conclusion must not depend on the arbitrary lag.
print("\n--- Diebold-Mariano (HAC): z vs max_lag ---")
dm_rows = []
for lag in [0, 5, 10, 20, 40]:
    se = hac_se(d, max_lag=lag)
    z = dbar / se
    p = two_sided_p(z)
    dm_rows.append(dict(max_lag=lag, se=se, z=z, p_two_sided=p))
    print(f"  max_lag={lag:2d}  SE={se:.8f}  z={z:8.3f}  p={p:.3g}")

L = 20
B = 5000
print(f"\n--- moving-block bootstrap (L={L}, B={B}) on mean(d) ---")
boot = block_bootstrap(d, L, B)
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"block bootstrap 95% CI of mean(d): [{lo:.8f}, {hi:.8f}]  "
      f"(excludes 0 -> {'YES' if lo > 0 else 'NO'})")

r = float(np.corrcoef(pred, y)[0, 1])
zr = np.arctanh(r)
se_r = 1.0 / np.sqrt(n - 3)
lo_r, hi_r = np.tanh(zr - 1.96 * se_r), np.tanh(zr + 1.96 * se_r)
p_r = two_sided_p(zr / se_r)
print(f"\nPearson r = {r:.4f}  95% CI = [{lo_r:.4f}, {hi_r:.4f}]  "
      f"z = {zr/se_r:.1f}  p = {p_r:.3g}")

# ---------------- Part B: seed sensitivity on fold 5 ----------------
print(f"\n=== Part B: seed sensitivity (train 0-63 -> val 64-70, @{NUM_ROUNDS}, num_threads=1) ===")
tr = X[X["month"] <= 63]
va = X[X["month"].isin([64, 65, 66, 67, 68, 69, 70])]
y5 = va["target"].to_numpy()
base0_5 = float(np.sqrt(np.mean(y5 ** 2)))
print(f"baseline(zero-pred) RMSE = {base0_5:.6f}")

seeds = [42, 0, 1, 7, 123, 777, 2024, 99, 55, 888,
         314, 2718, 1618, 1024, 2048, 4096, 17, 23, 31, 97]
rows = []
for sd in seeds:
    p = dict(params)
    p["seed"] = sd
    m = lgb.train(p, lgb.Dataset(tr[feature_cols], tr["target"]), num_boost_round=NUM_ROUNDS)
    pr = m.predict(va[feature_cols], num_iteration=NUM_ROUNDS)
    rm = float(np.sqrt(np.mean((pr - y5) ** 2)))
    pear = float(np.corrcoef(pr, y5)[0, 1])
    rows.append(dict(seed=sd, rmse=rm, ratio=rm / base0_5, pearson=pear))
    print(f"  seed={sd:5d}  RMSE={rm:.6f}  RMSE/base0={rm/base0_5:.4f}  Pearson={pear:.4f}")

res = pd.DataFrame(rows)
worst = res.loc[res["ratio"].idxmax()]
print(f"\nall {len(res)} seeds beat baseline: {'YES' if (res['ratio'] < 1).all() else 'NO'}")
print(f"ratio range: [{res['ratio'].min():.4f}, {res['ratio'].max():.4f}]  "
      f"worst seed={int(worst['seed'])}")

# ---------------- persist everything ----------------
res.to_csv(str(REPO_DIR / "significance_results.csv"), index=False)

summary_rows = [
    ("pooled_n", n),
    ("rmse", rmse),
    ("baseline_zero_pred_rmse", base0),
    ("rel_improvement_pct", rel * 100),
    ("mean_loss_diff_d", dbar),
    ("pearson_r", r),
    ("pearson_p_two_sided", p_r),
    ("pearson_ci_lo", lo_r),
    ("pearson_ci_hi", hi_r),
    ("block_bootstrap_L", L),
    ("block_bootstrap_B", B),
    ("block_bootstrap_ci_lo", lo),
    ("block_bootstrap_ci_hi", hi),
]
for r_ in dm_rows:
    summary_rows.append((f"dm_z_maxlag{r_['max_lag']}", r_["z"]))
    summary_rows.append((f"dm_p_maxlag{r_['max_lag']}", r_["p_two_sided"]))
summary_rows += [
    ("n_seeds", len(res)),
    ("seed_ratio_min", res["ratio"].min()),
    ("seed_ratio_max", res["ratio"].max()),
    ("all_seeds_beat_baseline", float((res["ratio"] < 1).all())),
]
pd.DataFrame(summary_rows, columns=["metric", "value"]).to_csv(
    str(REPO_DIR / "significance_summary.csv"), index=False)
print("\n[saved] significance_results.csv  significance_summary.csv")