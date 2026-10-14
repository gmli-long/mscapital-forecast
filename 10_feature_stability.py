"""Feature importance stability analysis + feature pruning.

Phase A: 6-fold fixed-155 walk-forward, collect per-fold gain importance.
Phase B: normalize gain to share, measure cross-fold stability (top-20 frequency,
         CV of share), pick features by cumulative mean-share <= 95%.
Phase C: retrain the same 6 folds with the pruned feature set, compare RMSE/pearson
         against the full-feature baseline (same fixed rounds, no early stop).
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import time

from config import REPO_DIR
from common import PARAMS, FOLDS, NUM_ROUNDS, load_X

OUT_STAB = str(REPO_DIR / "feature_stability.csv")

CUM_THRESH = 0.95
TOP_K = 20

X, feature_cols = load_X()

params = dict(PARAMS)
folds = FOLDS


def run_folds(feats_cols, collect_imp=False):
    imp_shares = {c: [] for c in feats_cols}
    top_sets = []
    metrics = []
    for tr_hi, val_ms in folds:
        tr = X[X["month"] <= tr_hi]
        va = X[X["month"].isin(val_ms)]
        dtr = lgb.Dataset(tr[feats_cols], tr["target"])
        model = lgb.train(params, dtr, num_boost_round=NUM_ROUNDS)
        pred = model.predict(va[feats_cols], num_iteration=NUM_ROUNDS)
        y = va["target"].to_numpy()
        metrics.append(dict(
            rmse=float(np.sqrt(np.mean((pred - y) ** 2))),
            pearson=float(np.corrcoef(pred, y)[0, 1]),
        ))
        if collect_imp:
            gain = model.feature_importance("gain").astype(float)
            share = gain / gain.sum()
            for c, s in zip(feats_cols, share):
                imp_shares[c].append(s)
            top_sets.append(set(np.array(feats_cols)[np.argsort(-gain)[:TOP_K]]))
    return metrics, imp_shares, top_sets


t0 = time.time()
print(f"Phase A/B: full-feature {len(feature_cols)} cols, fixed {NUM_ROUNDS} ...")
full_metrics, imp_shares, top_sets = run_folds(feature_cols, collect_imp=True)

# --- stability table ---
stab = []
for c in feature_cols:
    shares = np.array(imp_shares[c])
    stab.append(dict(
        feature=c,
        mean_share=shares.mean(),
        std_share=shares.std(),
        cv=shares.std() / shares.mean() if shares.mean() > 0 else np.nan,
        top20_freq=sum(c in s for s in top_sets),
    ))
stab = pd.DataFrame(stab).sort_values("mean_share", ascending=False).reset_index(drop=True)
stab.to_csv(OUT_STAB, index=False)
print("[saved]", OUT_STAB)

print("\n=== top 25 features by mean share ===")
show = stab.head(25).copy()
show["mean_share%"] = (show["mean_share"] * 100).round(2)
show["cv"] = show["cv"].round(2)
print(show[["feature", "mean_share%", "top20_freq", "cv"]].to_string(index=False))

# --- selection: cumulative mean share <= threshold ---
stab2 = stab  # already sorted
cum = stab2["mean_share"].cumsum()
n_keep = int((cum <= CUM_THRESH).sum()) + 1  # include the feature crossing the threshold
kept = stab2["feature"].iloc[:n_keep].tolist()
dropped = [c for c in feature_cols if c not in kept]
print(f"\n=== selection (cum mean-share <= {CUM_THRESH*100:.0f}%) ===")
print(f"kept {len(kept)} / {len(feature_cols)} features, dropped {len(dropped)}")
print("dropped (long-tail noise):", dropped)

# --- Phase C: pruned CV ---
print(f"\nPhase C: pruned-feature CV {len(kept)} cols ...")
pruned_metrics, _, _ = run_folds(kept, collect_imp=False)

full = pd.DataFrame(full_metrics)
prn = pd.DataFrame(pruned_metrics)
print("\n=== full vs pruned (fixed 155, per-fold) ===")
for i, (f, p) in enumerate(zip(full_metrics, pruned_metrics)):
    print(f"  fold {i}  val {folds[i][1][0]}-{folds[i][1][-1]}:  full RMSE={f['rmse']:.6f} r={f['pearson']:.4f} | "
          f"pruned RMSE={p['rmse']:.6f} r={p['pearson']:.4f}")

print("\n=== summary ===")
print(f"full   : RMSE {full.rmse.mean():.6f} ± {full.rmse.std():.6f}   Pearson {full.pearson.mean():.5f}")
print(f"pruned : RMSE {prn.rmse.mean():.6f} ± {prn.rmse.std():.6f}   Pearson {prn.pearson.mean():.5f}")
print(f"delta  : RMSE {(prn.rmse.mean()-full.rmse.mean()):+.6f}  "
      f"Pearson {(prn.pearson.mean()-full.pearson.mean()):+.5f}")
print(f"\ntotal {time.time()-t0:.1f}s")

# save pruned feature list
with open(REPO_DIR / "pruned_features.txt", "w") as fh:
    fh.write("\n".join(kept))
print("[saved] pruned_features.txt")